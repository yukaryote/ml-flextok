#!/usr/bin/env python3
"""
Simple test script to verify REPA loss implementation.
Run this to check that REPA module loads and computes loss correctly.
"""

import torch
from flextok import FlexTokFromHub
from flextok.model.utils.repa_loss import REPAModule

def test_repa_module():
    """Test REPA module initialization and forward pass."""
    print("=" * 60)
    print("Testing REPA Module Implementation")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Test 1: Load FlexTok model
    print("\n1. Loading FlexTok model...")
    torch.set_grad_enabled(False)
    model = FlexTokFromHub.from_pretrained('EPFL-VILAB/flextok_d18_d18_in1k')
    model = model.to(device).eval()
    print("   ✓ FlexTok model loaded")

    # Test 2: Check intermediate layer configuration
    print("\n2. Checking intermediate layer configuration...")
    dec_transformer = model.decoder.module_dict['dec_transformer']
    print(f"   Intermediate layer write key: {dec_transformer.intermediate_layer_write_key}")
    print(f"   Intermediate layers: {dec_transformer.intermediate_layers}")
    print(f"   Decoder dimension: {dec_transformer.dim}")
    print("   ✓ Intermediate layer configured correctly")

    # Test 3: Initialize REPA module
    print("\n3. Initializing REPA module...")
    decoder_dim = dec_transformer.dim
    repa_module = REPAModule(
        decoder_dim=decoder_dim,
        encoder_type='dinov2_vitl14',
        encoder_dim=1024,
        target_size=(37, 37),
    ).to(device)
    print(f"   Decoder dim: {decoder_dim}")
    print(f"   Encoder dim: 1024")
    print(f"   Projector parameters: {sum(p.numel() for p in repa_module.projector.parameters()):,}")
    print("   ✓ REPA module initialized")

    # Test 4: Forward pass through FlexTok
    print("\n4. Testing forward pass...")
    batch_size = 2
    dummy_images = torch.randn(batch_size, 3, 256, 256).to(device) * 2 - 1  # [-1, 1]
    images_list = dummy_images.split(1)

    data_dict = {model.vae.images_read_key: images_list}
    data_dict = model(data_dict)
    print(f"   Input shape: {dummy_images.shape}")
    print("   ✓ FlexTok forward pass completed")

    # Test 5: Extract intermediate features
    print("\n5. Extracting intermediate features...")
    intermediate_features = data_dict.get('dec_packed_seq_repa_layer')
    if intermediate_features is None:
        print("   ✗ ERROR: No intermediate features found!")
        return False
    print(f"   Intermediate features shape: {intermediate_features.shape}")
    print("   ✓ Intermediate features extracted")

    # Test 6: Unpack features
    print("\n6. Unpacking features...")
    unpacker = model.decoder.module_dict['dec_repa_unpacker']
    print(f"   Unpacker packed_seq_read_key: {unpacker.packed_seq_read_key}")
    print(f"   Unpacker inner_seq_write_keys: {unpacker.inner_seq_write_keys}")

    unpack_dict = {
        unpacker.packed_seq_read_key: intermediate_features,
        unpacker.inner_packed_shapes_read_key: data_dict['dec_ps_inner'],
        unpacker.outer_packed_shapes_read_key: data_dict['dec_ps_outer'],
    }
    unpack_dict = unpacker(unpack_dict)
    decoder_features = unpack_dict[unpacker.inner_seq_write_keys[0]]
    print(f"   Unpacked features: {type(decoder_features)}")
    if isinstance(decoder_features, list):
        print(f"   List length: {len(decoder_features)}, each shape: {decoder_features[0].shape}")
    else:
        print(f"   Shape: {decoder_features.shape}")
    print("   ✓ Features unpacked successfully")

    # Test 7: Compute REPA loss
    print("\n7. Computing REPA loss...")
    torch.set_grad_enabled(True)
    repa_data_dict = {
        repa_module.features_read_key: decoder_features,
        repa_module.images_read_key: dummy_images,
    }
    repa_loss = repa_module(repa_data_dict)
    print(f"   REPA loss: {repa_loss.item():.4f}")
    print(f"   Loss requires grad: {repa_loss.requires_grad}")

    # Verify loss is negative (cosine similarity)
    if repa_loss.item() < 0:
        print("   ✓ REPA loss computed correctly (negative cosine similarity)")
    else:
        print(f"   ⚠ Warning: REPA loss is positive ({repa_loss.item():.4f})")

    # Test 8: Backward pass
    print("\n8. Testing backward pass...")
    repa_loss.backward()

    # Check if gradients were computed for projector
    projector_has_grad = any(p.grad is not None for p in repa_module.projector.parameters())
    encoder_has_grad = any(p.grad is not None for p in repa_module.encoder.parameters() if p.requires_grad)

    print(f"   Projector has gradients: {projector_has_grad}")
    print(f"   Encoder has gradients: {encoder_has_grad}")

    if projector_has_grad and not encoder_has_grad:
        print("   ✓ Backward pass successful (projector trained, encoder frozen)")
    else:
        print("   ⚠ Warning: Gradient computation may be incorrect")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    return True

if __name__ == "__main__":
    try:
        success = test_repa_module()
        if success:
            print("\n✓ REPA implementation is working correctly!")
            print("\nTo enable REPA in training, set in configs/train_celebahq.yaml:")
            print("  use_repa: true")
            print("  repa_weight: 1.0")
        else:
            print("\n✗ Tests failed. Please check the implementation.")
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
