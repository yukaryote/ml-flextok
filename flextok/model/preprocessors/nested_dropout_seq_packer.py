# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. and EPFL. All Rights Reserved.
"""
Modified BlockWiseSequencePacker that integrates nested dropout attention masking.

This packer creates attention masks that respect both the block-wise structure
(different sequences don't attend to each other) AND the nested dropout masking
(tokens beyond keep_k are masked out).
"""

import math
from functools import lru_cache, partial
from typing import Any, Dict, List, Optional

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, create_mask

from .attention_masked_nested_dropout import generate_nested_dropout_attention_mask
from .flex_seq_packing import (
    create_block_mask_cached,
    strict_zip,
    next_highest_multiple,
    expand_emb,
    expand_emb_per_subseq,
    generate_seq_ids,
)

__all__ = ["NestedDropoutSequencePacker"]


class NestedDropoutSequencePacker(nn.Module):
    """
    Sequence packer that integrates nested dropout attention masking.

    This module combines the functionality of BlockWiseSequencePacker with
    nested dropout masking. It reads keep_k values from the data_dict and
    creates attention masks that prevent masked tokens from participating
    in attention.

    Args:
        input_list_read_keys: List of keys to read input lists of tensors from.
        packed_seq_write_key: Key to write the packed sequence into.
        block_mask_write_key: Key to write the block-wise attention mask into.
        inner_packed_shapes_write_key: Key to write the packed shapes of inner sequences.
        outer_packed_shapes_write_key: Key to write the packed shape of outer sequence.
        keep_k_read_key: Key to read keep_k values from (set by AttentionMaskedNestedDropout).
        mask_mode: Block-wise attention mask mode ('full', 'causal', or 'causal_last').
        max_seq_len: Optionally pads packed sequence to this length.
        pad_to_multiple: Pads sequence to next multiple of this value (e.g., 128 for FlexAttention).
        emb_packing_fn_write_key: Optional embedding packing function key.
        per_subseq_embs: If True, applies emb_packing_fn to each subsequence.
        compile_block_mask: Whether to compile FlexAttention's create_block_mask.
        return_materialized_mask: If True, returns materialized mask instead of block mask.
    """

    def __init__(
        self,
        input_list_read_keys: List[str],
        packed_seq_write_key: str,
        block_mask_write_key: str,
        inner_packed_shapes_write_key: str,
        outer_packed_shapes_write_key: str,
        keep_k_read_key: str = "nested_dropout_keep_k",
        mask_mode: str = "full",
        max_seq_len: Optional[int] = None,
        pad_to_multiple: Optional[int] = None,
        emb_packing_fn_write_key: Optional[str] = None,
        per_subseq_embs: bool = False,
        compile_block_mask: bool = True,
        return_materialized_mask: bool = False,
    ):
        super().__init__()
        self.input_list_read_keys = input_list_read_keys
        self.packed_seq_write_key = packed_seq_write_key
        self.block_mask_write_key = block_mask_write_key
        self.inner_packed_shapes_write_key = inner_packed_shapes_write_key
        self.outer_packed_shapes_write_key = outer_packed_shapes_write_key
        self.keep_k_read_key = keep_k_read_key
        self.emb_packing_fn_write_key = emb_packing_fn_write_key

        self.mask_mode = mask_mode
        self.max_seq_len = max_seq_len
        self.pad_to_multiple = pad_to_multiple
        if max_seq_len is not None and pad_to_multiple is not None:
            raise ValueError("Only one of max_seq_len or pad_to_multiple should be provided.")
        self.per_subseq_embs = per_subseq_embs

        self.compile_block_mask = compile_block_mask
        self.create_block_mask = torch.compiler.disable(create_block_mask_cached)
        self.return_materialized_mask = return_materialized_mask

        # For pretty printing
        self._init_args = locals().copy()
        self._init_args.pop("self")
        self._init_args.pop("__class__")

    def __repr__(self):
        cls_name = self.__class__.__name__
        args_str = ",\n  ".join(f"{k}={v!r}" for k, v in self._init_args.items())
        return f"{cls_name}(\n  {args_str}\n)"

    @torch.compiler.disable
    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        # From the data_dict, get the lists containing the aligned tensors
        list_of_tensor_lists = [data_dict[key] for key in self.input_list_read_keys]

        # Concatenate each sample across the lists
        tensors_concat_list = []
        packed_shapes_list = []
        for tensors in strict_zip(*list_of_tensor_lists):
            sample_packed, ps = einops.pack(tensors, "b * d")
            tensors_concat_list.append(sample_packed)
            packed_shapes_list.append(ps)

        # Pack tensors into one large sequence
        tensors_packed, ps = einops.pack(tensors_concat_list, "b * d")
        B, N_orig, D = tensors_packed.shape

        # Only supporting B=1 until https://github.com/pytorch/pytorch/issues/134560 is resolved
        assert B == 1

        device = str(tensors_packed.device)

        # Determine max_seq_len for padding
        if self.pad_to_multiple is not None:
            max_seq_len = next_highest_multiple(N_orig, self.pad_to_multiple)
        else:
            max_seq_len = self.max_seq_len

        # Check if nested dropout is active (keep_k values provided)
        if self.keep_k_read_key in data_dict:
            keep_k_list = data_dict[self.keep_k_read_key]

            # Calculate num_sequences_per_sample based on input_list_read_keys
            # E.g., if input_list_read_keys = [dec_patches, dec_quants], then num_sequences_per_sample = 2
            num_sequences_per_sample = len(self.input_list_read_keys)

            # Generate nested dropout mask that respects keep_k
            if self.mask_mode == "full":
                mask_fn, mask_ids = generate_nested_dropout_attention_mask(
                    tuple(ps),
                    keep_k_list,
                    max_seq_len=max_seq_len,
                    device=device,
                    num_sequences_per_sample=num_sequences_per_sample,
                )
            else:
                # For causal modes, we'd need to combine nested dropout with causality
                # For now, fall back to standard masking (can be extended later)
                raise NotImplementedError(
                    f"Nested dropout with mask_mode='{self.mask_mode}' not yet implemented. Use 'full'."
                )
        else:
            # No nested dropout - use standard packed sequence masking
            from .flex_seq_packing import generate_packed_seq_mask, generate_causal_packed_seq_mask, generate_prefix_packed_seq_mask

            if self.mask_mode == "full":
                mask_fn, seq_ids = generate_packed_seq_mask(
                    tuple(ps),
                    max_seq_len=max_seq_len,
                    device=device,
                )
            elif self.mask_mode == "causal":
                mask_fn, seq_ids = generate_causal_packed_seq_mask(
                    tuple(ps),
                    max_seq_len=max_seq_len,
                    device=device,
                )
            elif self.mask_mode == "causal_last":
                prefix_lens = [
                    sum([shape.numel() for shape in ps_i[:-1]]) for ps_i in packed_shapes_list
                ]
                mask_fn, seq_ids = generate_prefix_packed_seq_mask(
                    tuple(ps),
                    tuple(prefix_lens),
                    max_seq_len=max_seq_len,
                    device=device,
                )
            else:
                raise ValueError(f"Invalid mask mode {self.mask_mode}")

        N = max_seq_len if max_seq_len is not None else N_orig
        assert (
            N % 128 == 0
        ), f"flex_attention sequence length must be a multiple of 128, but current is {N}."

        if self.return_materialized_mask:
            block_mask = create_mask(mask_fn, None, None, N, N, device=device)
        else:
            block_mask = self.create_block_mask(
                mask_fn, None, None, N, N, device=device, _compile=self.compile_block_mask
            )

        # Optionally zero-pad packed sequence
        num_padding_tokens = N - N_orig
        if num_padding_tokens > 0:
            tensors_packed = F.pad(tensors_packed, (0, 0, 0, num_padding_tokens))

        # Optionally add embedding packing function
        if self.emb_packing_fn_write_key is not None:
            seq_lens = [shape.numel() for shape in ps]
            if self.per_subseq_embs:
                emb_packing_fn = partial(
                    expand_emb_per_subseq, packed_shapes_list=packed_shapes_list
                )
            else:
                emb_packing_fn = partial(expand_emb, seq_lens=seq_lens)
            data_dict[self.emb_packing_fn_write_key] = emb_packing_fn

        data_dict[self.packed_seq_write_key] = tensors_packed
        data_dict[self.block_mask_write_key] = block_mask
        data_dict[self.inner_packed_shapes_write_key] = packed_shapes_list
        data_dict[self.outer_packed_shapes_write_key] = ps

        return data_dict
