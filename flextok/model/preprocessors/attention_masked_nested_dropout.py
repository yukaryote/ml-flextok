# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. and EPFL. All Rights Reserved.
"""
Attention-masked nested dropout for FlexTok.

IMPORTANT: This module applies nested dropout masking ONLY to register tokens,
NOT to image patches. When sequences are packed as [patches, registers], only
the register tokens beyond keep_k are masked out in attention.

Example with keep_k=64:
    Packed sequence: [p₀...p₁₀₂₃, r₀...r₆₃, r₆₄...r₂₅₅]
                      ←─patches─→  ←kept→  ←─masked─→

    - All 1024 patches: fully attended (never masked)
    - First 64 registers: fully attended (kept)
    - Last 192 registers: masked out (cannot attend or be attended to)
"""
from functools import lru_cache
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .registers import is_power_of_two, powers_of_two

__all__ = ["AttentionMaskedNestedDropout"]


@lru_cache
def generate_nested_dropout_mask_ids(ps, keep_k_list, max_seq_len=None, device="cuda", masked_id=-2, num_sequences_per_sample=2):
    """
    Generate mask IDs for nested dropout, where tokens beyond keep_k are marked as masked.

    IMPORTANT: This applies masking ONLY to the LAST subsequence in each packed sample
    (i.e., the register tokens), NOT to the image patches.

    Args:
        ps: Packed shapes (tuple of shape objects)
            Example: [patches_img0, registers_img0, patches_img1, registers_img1, ...]
        keep_k_list: List of keep_k values for each SAMPLE (not each subsequence)
            Example: [64, 128] means keep 64 registers for img0, 128 for img1
        max_seq_len: Optional maximum sequence length for padding
        masked_id: ID to use for masked tokens (default: -2, different from padding -1)
        device: Device to create tensors on
        num_sequences_per_sample: Number of subsequences per sample (default: 2 for [patches, registers])

    Returns:
        mask_ids: Tensor where each position is marked with either its sequence ID or masked_id
    """
    mask_ids = []

    # Process in groups of num_sequences_per_sample
    for sample_idx in range(len(ps) // num_sequences_per_sample):
        keep_k = keep_k_list[sample_idx] if sample_idx < len(keep_k_list) else None

        # Get the subsequences for this sample
        start_idx = sample_idx * num_sequences_per_sample
        subseqs_shapes = ps[start_idx:start_idx + num_sequences_per_sample]

        for subseq_idx, shape in enumerate(subseqs_shapes):
            seq_len = shape.numel()
            seq_id = start_idx + subseq_idx

            # Create mask IDs for this subsequence
            seq_mask_ids = torch.full((seq_len,), fill_value=seq_id, device=device)

            # ONLY apply nested dropout to the LAST subsequence (register tokens)
            is_last_subseq = (subseq_idx == num_sequences_per_sample - 1)
            if is_last_subseq and keep_k is not None and keep_k < seq_len:
                # Mask tokens beyond keep_k in register tokens
                seq_mask_ids[keep_k:] = masked_id

            mask_ids.append(seq_mask_ids)

    if mask_ids:
        mask_ids = torch.cat(mask_ids)
    else:
        mask_ids = torch.empty((0,), dtype=torch.long, device=device)

    if max_seq_len is not None:
        seq_len = len(mask_ids)
        assert max_seq_len >= seq_len
        # Pad with -1 (different from masked_id)
        mask_ids = F.pad(mask_ids, (0, max_seq_len - seq_len), mode="constant", value=-1)

    return mask_ids


def generate_nested_dropout_attention_mask(ps, keep_k_list, max_seq_len=None, device="cuda", masked_id=-2, num_sequences_per_sample=2):
    """
    Generate FlexAttention mask function for nested dropout.

    Masking rules:
    - Tokens within the same sequence and both within keep_k can attend to each other
    - Masked tokens (beyond keep_k in REGISTER TOKENS ONLY) cannot attend or be attended to
    - Padding tokens cannot attend or be attended to
    - Different sequences cannot attend to each other
    - Image patches are NEVER masked (full attention to all patches)

    Args:
        ps: Packed shapes (tuple of shape objects)
        keep_k_list: List of keep_k values for each sample
        max_seq_len: Optional maximum sequence length for padding
        device: Device to create tensors on
        masked_id: ID to use for masked tokens
        num_sequences_per_sample: Number of subsequences per sample (e.g., 2 for [patches, registers])
    """
    mask_ids = generate_nested_dropout_mask_ids(
        ps, tuple(keep_k_list), max_seq_len=max_seq_len, device=device, masked_id=masked_id,
        num_sequences_per_sample=num_sequences_per_sample
    )

    def nested_dropout_masking(b, h, q_idx, kv_idx):
        q_mask_id = mask_ids[q_idx]
        kv_mask_id = mask_ids[kv_idx]

        # Both must not be masked or padded
        not_masked_q = q_mask_id >= 0
        not_masked_kv = kv_mask_id >= 0

        # Must be from the same sequence (same mask_id when both are not masked)
        same_seq_mask = q_mask_id == kv_mask_id

        return same_seq_mask & not_masked_q & not_masked_kv

    return nested_dropout_masking, mask_ids


class AttentionMaskedNestedDropout(nn.Module):
    """
    Module that randomly drops tokens using attention masking instead of a learnable mask token.

    This module creates a custom attention mask for the decoder transformer that prevents
    masked tokens from being attended to or attending to anything. This is more memory
    efficient than the learnable mask token approach and ensures masked tokens are truly
    "dropped" from the computation.

    Args:
        read_key: Key to read the input tokens from.
        keep_k_write_key: Key to write the keep_k values to (used for mask generation).
        eval_keep_k_read_key: During inference, read the number of tokens to keep from this key.
        train_keep_k_write_key: During training, write the sampled keep_k to this key.
        size_sampling_mode: Method to sample the number of tokens to randomly drop.
        block_mask_write_key: Key to write the FlexAttention block mask to.
    """

    def __init__(
        self,
        read_write_key: str,
        keep_k_write_key: str = "nested_dropout_keep_k",
        eval_keep_k_read_key: Optional[str] = "eval_keep_k",
        train_keep_k_write_key: Optional[str] = "train_keep_k",
        size_sampling_mode: Literal["uniform", "pow2", "uniform_pow2"] = "uniform",
        block_mask_write_key: str = "nested_dropout_block_mask",
    ):
        super().__init__()
        self.read_write_key = read_write_key
        self.keep_k_write_key = keep_k_write_key
        self.eval_keep_k_read_key = eval_keep_k_read_key
        self.train_keep_k_write_key = train_keep_k_write_key
        self.size_sampling_mode = size_sampling_mode
        self.block_mask_write_key = block_mask_write_key

        # No learnable mask token parameter!

    def sample_keep_k(self, N):
        if self.size_sampling_mode == "uniform":
            keep_k = np.random.randint(low=1, high=N + 1)
        elif self.size_sampling_mode == "pow2":
            assert is_power_of_two(N)
            keep_k = np.random.choice(powers_of_two(1, N))
        elif self.size_sampling_mode == "uniform_pow2":
            k = np.random.randint(low=1, high=N + 1)
            keep_k = k if is_power_of_two(k) else 1 << k.bit_length()
        else:
            raise ValueError(f"size_sampling_mode {self.size_sampling_mode} is not defined.")
        return keep_k

    @torch.compiler.disable
    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward pass that generates keep_k values but does NOT modify tokens.
        Instead, it writes the keep_k values to the data_dict which will be used
        by the sequence packer to create the attention mask.
        """
        if not self.training:
            # Inference mode: use provided keep_k or keep all tokens
            if self.eval_keep_k_read_key is None or self.eval_keep_k_read_key not in data_dict:
                # No dropout during inference if eval_keep_k not provided
                return data_dict

            keep_ks = data_dict[self.eval_keep_k_read_key]
        else:
            # Training mode: sample random keep_k for each sequence
            keep_ks = []
            for i in range(len(data_dict[self.read_write_key])):
                N = data_dict[self.read_write_key][i].shape[1]
                keep_k = self.sample_keep_k(N)
                keep_ks.append(keep_k)

            # Write keep_k for logging/debugging
            data_dict[self.train_keep_k_write_key] = keep_ks

        # Write keep_k values to be used by the sequence packer
        data_dict[self.keep_k_write_key] = keep_ks

        return data_dict
