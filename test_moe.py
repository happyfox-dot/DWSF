#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple test script to verify MoE decoder implementation
"""

import torch
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from networks.models.Decoder import Decoder
from networks.models.DecoderMoE import DecoderMoE
from networks.models.EncoderDecoder import EncoderDecoder
from networks.models.Encoder import Encoder


def test_moe_decoder():
    """Test standalone MoE decoder"""
    print("=" * 50)
    print("Testing MoE Decoder")
    print("=" * 50)
    
    # Parameters
    batch_size = 4
    channels = 3
    height = 128
    width = 128
    message_length = 30
    
    # Create a sample input (watermarked image)
    sample_input = torch.randn(batch_size, channels, height, width)
    
    # Test original decoder
    print("\n1. Testing original Decoder (use_moe=False)...")
    decoder_original = Decoder(message_length=message_length, use_moe=False)
    decoder_original.eval()
    
    with torch.no_grad():
        output_original = decoder_original(sample_input)
    
    print(f"   Input shape: {sample_input.shape}")
    print(f"   Output shape: {output_original.shape}")
    print(f"   Expected shape: ({batch_size}, {message_length})")
    assert output_original.shape == (batch_size, message_length), "Original decoder output shape mismatch!"
    print("   ✓ Original decoder test passed!")
    
    # Test MoE decoder
    print("\n2. Testing MoE Decoder (use_moe=True)...")
    decoder_moe = Decoder(message_length=message_length, use_moe=True, num_experts=4)
    decoder_moe.eval()
    
    with torch.no_grad():
        output_moe = decoder_moe(sample_input)
    
    print(f"   Input shape: {sample_input.shape}")
    print(f"   Output shape: {output_moe.shape}")
    print(f"   Expected shape: ({batch_size}, {message_length})")
    assert output_moe.shape == (batch_size, message_length), "MoE decoder output shape mismatch!"
    print("   ✓ MoE decoder test passed!")
    
    # Test standalone DecoderMoE class
    print("\n3. Testing standalone DecoderMoE class...")
    decoder_moe_standalone = DecoderMoE(message_length=message_length, num_experts=4)
    decoder_moe_standalone.eval()
    
    with torch.no_grad():
        output_moe_standalone = decoder_moe_standalone(sample_input)
    
    print(f"   Input shape: {sample_input.shape}")
    print(f"   Output shape: {output_moe_standalone.shape}")
    print(f"   Expected shape: ({batch_size}, {message_length})")
    assert output_moe_standalone.shape == (batch_size, message_length), "Standalone MoE decoder output shape mismatch!"
    print("   ✓ Standalone MoE decoder test passed!")
    
    print("\n" + "=" * 50)
    print("All MoE decoder tests passed! ✓")
    print("=" * 50)


def test_encoder_decoder_moe():
    """Test EncoderDecoder with MoE (simplified without noise layers)"""
    print("\n" + "=" * 50)
    print("Testing Decoder Integration in EncoderDecoder")
    print("=" * 50)
    
    # Parameters
    batch_size = 2
    channels = 3
    height = 128
    width = 128
    message_length = 30
    
    # Create sample inputs
    sample_image = torch.randn(batch_size, channels, height, width)
    sample_message = torch.randint(0, 2, (batch_size, message_length)).float()
    
    # Test decoder integration by directly calling encoder and decoder
    print("\n1. Testing Decoder (original) in pipeline...")
    from networks.models.Encoder import Encoder
    
    encoder = Encoder(H=height, W=width, message_length=message_length)
    decoder_original = Decoder(message_length=message_length, use_moe=False)
    encoder.eval()
    decoder_original.eval()
    
    with torch.no_grad():
        encoded_img = encoder(sample_image, sample_message)
        decoded_msg = decoder_original(encoded_img)
    
    print(f"   Input image shape: {sample_image.shape}")
    print(f"   Input message shape: {sample_message.shape}")
    print(f"   Encoded image shape: {encoded_img.shape}")
    print(f"   Decoded message shape: {decoded_msg.shape}")
    assert encoded_img.shape == sample_image.shape, "Encoded image shape mismatch!"
    assert decoded_msg.shape == sample_message.shape, "Decoded message shape mismatch!"
    print("   ✓ Original decoder integration test passed!")
    
    # Test MoE decoder integration
    print("\n2. Testing Decoder (MoE) in pipeline...")
    decoder_moe = Decoder(message_length=message_length, use_moe=True, num_experts=4)
    decoder_moe.eval()
    
    with torch.no_grad():
        decoded_msg_moe = decoder_moe(encoded_img)
    
    print(f"   Input encoded image shape: {encoded_img.shape}")
    print(f"   Decoded message shape: {decoded_msg_moe.shape}")
    assert decoded_msg_moe.shape == sample_message.shape, "MoE decoded message shape mismatch!"
    print("   ✓ MoE decoder integration test passed!")
    
    print("\n" + "=" * 50)
    print("All integration tests passed! ✓")
    print("=" * 50)


def test_parameter_count():
    """Compare parameter counts between original and MoE decoders"""
    print("\n" + "=" * 50)
    print("Parameter Count Comparison")
    print("=" * 50)
    
    message_length = 30
    
    # Original decoder
    decoder_original = Decoder(message_length=message_length, use_moe=False)
    params_original = sum(p.numel() for p in decoder_original.parameters())
    
    # MoE decoder
    decoder_moe = Decoder(message_length=message_length, use_moe=True, num_experts=4)
    params_moe = sum(p.numel() for p in decoder_moe.parameters())
    
    print(f"\n   Original Decoder Parameters: {params_original:,}")
    print(f"   MoE Decoder Parameters: {params_moe:,}")
    print(f"   Increase: {params_moe - params_original:,} ({(params_moe/params_original - 1)*100:.2f}%)")
    
    print("\n" + "=" * 50)


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("MoE Decoder Implementation Test Suite")
    print("=" * 70)
    
    try:
        # Run all tests
        test_moe_decoder()
        test_encoder_decoder_moe()
        test_parameter_count()
        
        print("\n" + "=" * 70)
        print("ALL TESTS PASSED SUCCESSFULLY! ✓✓✓")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print("\n" + "=" * 70)
        print(f"TEST FAILED: {str(e)}")
        print("=" * 70 + "\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
