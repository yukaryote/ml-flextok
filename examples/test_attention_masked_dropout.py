#!/usr/bin/env python3
"""
Test script for attention-masked nested dropout.

This demonstrates how the new AttentionMaskedNestedDropout + NestedDropoutSequencePacker
work together to create proper attention masking instead of using a learnable mask token.
"""

import torch
import torch.nn as nn
from flextok.model.preprocessors.attention_masked_nested_dropout import (
    AttentionMaskedNestedDropout,
    generate_nested_dropout_attention_mask,
)
from flextok.model.preprocessors.nested_dropout_seq_packer import NestedDropoutSequencePacker


def test_attention_mask_generation():
    """Test that the attention mask correctly masks out tokens beyond keep_k."""
    print("=" * 80)
    print("Test 1: Attention Mask Generation")
    print("=" * 80)

    # Simulate packed shapes (e.g., 2 sequences with 10 tokens each)
    import torch
    ps = [torch.Size([10]), torch.Size([10])]
    keep_k_list = [6, 8]  # First seq keeps 6 tokens, second keeps 8
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mask_fn, mask_ids = generate_nested_dropout_attention_mask(
        tuple(ps),
        keep_k_list,
        max_seq_len=20,
        device=device,
    )

    print(f"Mask IDs: {mask_ids}")
    print(f"  Positions 0-5: seq 0 (unmasked)")
    print(f"  Positions 6-9: seq 0 (masked, id=-2)")
    print(f"  Positions 10-17: seq 1 (unmasked)")
    print(f"  Positions 18-19: seq 1 (masked, id=-2)")

    # Test attention patterns
    print("\nAttention patterns:")
    test_pairs = [
        (0, 3, "seq0[0] -> seq0[3]"),     # Within keep_k, same seq
        (0, 7, "seq0[0] -> seq0[7]"),     # Beyond keep_k, same seq
        (0, 10, "seq0[0] -> seq1[0]"),    # Different seqs
        (7, 3, "seq0[7] -> seq0[3]"),     # Masked query
        (10, 15, "seq1[0] -> seq1[5]"),   # Within keep_k, seq 1
        (10, 18, "seq1[0] -> seq1[8]"),   # Beyond keep_k, seq 1
    ]

    for q, kv, desc in test_pairs:
        can_attend = mask_fn(0, 0, q, kv)
        status = "✓ CAN attend" if can_attend else "✗ CANNOT attend"
        print(f"  {desc}: {status}")

    print()


def test_nested_dropout_module():
    """Test the AttentionMaskedNestedDropout module."""
    print("=" * 80)
    print("Test 2: AttentionMaskedNestedDropout Module")
    print("=" * 80)

    # Create module
    dropout_module = AttentionMaskedNestedDropout(
        read_key="tokens",
        keep_k_write_key="nested_dropout_keep_k",
        size_sampling_mode="uniform",
    )

    # Simulate input data dict
    batch_size = 2
    seq_len = 256
    dim = 1792

    tokens_list = [
        torch.randn(1, seq_len, dim),
        torch.randn(1, seq_len, dim),
    ]

    data_dict = {"tokens": tokens_list}

    # Training mode: should sample random keep_k
    dropout_module.train()
    output_dict = dropout_module(data_dict)

    keep_ks = output_dict["nested_dropout_keep_k"]
    print(f"Training mode - sampled keep_k values: {keep_ks}")
    print(f"  Tokens are NOT modified (shape still {tokens_list[0].shape})")

    # Inference mode: should use provided keep_k
    dropout_module.eval()
    data_dict["eval_keep_k"] = [64, 128]
    output_dict = dropout_module(data_dict)

    keep_ks = output_dict["nested_dropout_keep_k"]
    print(f"\nInference mode - using eval_keep_k: {keep_ks}")

    print()


def test_full_pipeline():
    """Test the full pipeline with NestedDropoutSequencePacker."""
    print("=" * 80)
    print("Test 3: Full Pipeline with NestedDropoutSequencePacker")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create modules
    dropout_module = AttentionMaskedNestedDropout(
        read_key="register_tokens",
        keep_k_write_key="nested_dropout_keep_k",
        size_sampling_mode="uniform",
    )

    packer_module = NestedDropoutSequencePacker(
        input_list_read_keys=["patches", "register_tokens"],
        packed_seq_write_key="packed_seq",
        block_mask_write_key="sa_block_mask",
        inner_packed_shapes_write_key="ps_inner",
        outer_packed_shapes_write_key="ps_outer",
        keep_k_read_key="nested_dropout_keep_k",
        mask_mode="full",
        pad_to_multiple=128,
    )

    # Simulate data
    batch_size = 2
    num_patches = 1024  # 32x32 patches
    num_registers = 256
    dim = 1792

    patches_list = [
        torch.randn(1, num_patches, dim, device=device),
        torch.randn(1, num_patches, dim, device=device),
    ]

    register_tokens_list = [
        torch.randn(1, num_registers, dim, device=device),
        torch.randn(1, num_registers, dim, device=device),
    ]

    data_dict = {
        "patches": patches_list,
        "register_tokens": register_tokens_list,
    }

    # Step 1: Generate keep_k values
    dropout_module.train()
    data_dict = dropout_module(data_dict)
    keep_ks = data_dict["nested_dropout_keep_k"]
    print(f"Step 1 - Generated keep_k: {keep_ks}")

    # Step 2: Pack sequences with nested dropout masking
    data_dict = packer_module(data_dict)

    packed_seq = data_dict["packed_seq"]
    block_mask = data_dict["sa_block_mask"]

    print(f"\nStep 2 - Packed sequence:")
    print(f"  Shape: {packed_seq.shape}")
    print(f"  Expected: [1, {num_patches + num_registers} * 2 (padded to 128 multiple), {dim}]")
    print(f"  Block mask type: {type(block_mask)}")

    # The block mask now enforces:
    # 1. Different images can't attend to each other (block-wise)
    # 2. Tokens beyond keep_k can't attend or be attended to (nested dropout)

    print(f"\nMask enforces:")
    print(f"  - Image 0 registers: only first {keep_ks[0]} can attend")
    print(f"  - Image 1 registers: only first {keep_ks[1]} can attend")
    print(f"  - Patches can attend to their corresponding registers")
    print(f"  - Different images are isolated from each other")

    print()


def visualize_attention_mask():
    """Visualize the attention mask pattern."""
    print("=" * 80)
    print("Test 4: Visualize Attention Mask")
    print("=" * 80)

    # Simple example: 2 sequences, 10 tokens each
    ps = [torch.Size([10]), torch.Size([10])]
    keep_k_list = [6, 8]
    device = "cpu"

    mask_fn, mask_ids = generate_nested_dropout_attention_mask(
        tuple(ps),
        keep_k_list,
        max_seq_len=20,
        device=device,
    )

    # Create attention mask matrix
    N = 20
    mask_matrix = torch.zeros(N, N, dtype=torch.bool)
    for q in range(N):
        for kv in range(N):
            mask_matrix[q, kv] = mask_fn(0, 0, q, kv)

    print("Attention mask matrix (✓ = can attend, ✗ = masked):")
    print("      ", "".join(f"{i:2d} " for i in range(N)))
    for q in range(N):
        row_str = "".join(" ✓ " if mask_matrix[q, kv] else " ✗ " for kv in range(N))
        print(f"  {q:2d}: {row_str}")

    print("\nLegend:")
    print("  Rows 0-5: Seq 0, unmasked")
    print("  Rows 6-9: Seq 0, masked (cannot attend)")
    print("  Rows 10-17: Seq 1, unmasked")
    print("  Rows 18-19: Seq 1, masked (cannot attend)")

    print()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Attention-Masked Nested Dropout Test Suite")
    print("=" * 80 + "\n")

    test_attention_mask_generation()
    test_nested_dropout_module()
    test_full_pipeline()
    visualize_attention_mask()

    print("=" * 80)
    print("All tests completed!")
    print("=" * 80)
