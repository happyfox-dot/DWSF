# Mixture of Experts (MOE) Decoder

## Overview

The MOE (Mixture of Experts) architecture has been added to the watermark decoder to improve feature extraction through adaptive expert routing. This enhancement allows the decoder to use multiple specialized "expert" networks and dynamically route inputs based on their characteristics.

## Architecture

### Key Components

1. **Expert Networks**: Multiple specialized convolutional networks that process inputs differently
2. **Gating Network**: An adaptive routing mechanism that determines how to weight each expert's contribution
3. **MoE Blocks**: Combinations of experts and gating that replace some SENet blocks in the decoder

### Design Choices

- **Number of Experts**: Default is 4 experts per MOE block (configurable)
- **Expert Blocks**: Each expert contains 2 convolutional layers
- **Integration**: MOE blocks are strategically placed throughout the decoder pipeline
- **Parameter Increase**: ~65% more parameters compared to the original decoder

## Usage

### Basic Usage

#### Using the Modified Decoder Class

The simplest way to use MOE is through the modified `Decoder` class with the `use_moe` flag:

```python
from networks.models.Decoder import Decoder

# Original decoder (backward compatible)
decoder = Decoder(message_length=30, use_moe=False)

# MOE-enhanced decoder
decoder_moe = Decoder(message_length=30, use_moe=True, num_experts=4)
```

#### Using the Standalone DecoderMoE Class

Alternatively, use the dedicated MOE decoder class:

```python
from networks.models.DecoderMoE import DecoderMoE

decoder = DecoderMoE(message_length=30, num_experts=4)
```

#### Integration with EncoderDecoder

For full pipeline integration:

```python
from networks.models.EncoderDecoder import EncoderDecoder

# Without MOE (original)
encoder_decoder = EncoderDecoder(
    H=128, W=128,
    message_length=30,
    noise_layers=["Combined([Identity()])"],
    use_moe=False
)

# With MOE
encoder_decoder_moe = EncoderDecoder(
    H=128, W=128,
    message_length=30,
    noise_layers=["Combined([Identity()])"],
    use_moe=True,
    num_experts=4
)
```

### Training

To train with MOE decoder, simply modify your training script:

```python
# In train_ed.py
encoder_decoder = EncoderDecoder(
    H=H, W=W, 
    message_length=message_length, 
    noise_layers=train_noise_layer,
    use_moe=True,  # Enable MOE
    num_experts=4   # Number of experts
)
```

### Evaluation

For evaluation, ensure the decoder is loaded with the same MOE configuration:

```python
# In evaluate.py
encoder_decoder = EncoderDecoder(
    H=H, W=W, 
    message_length=message_length, 
    noise_layers=default_noise_layer,
    use_moe=True,  # Must match training configuration
    num_experts=4
)

# Load trained weights
encoder_decoder.encoder.load_state_dict(torch.load(args.pth_path+'/encoder_best.pth'))
encoder_decoder.decoder.load_state_dict(torch.load(args.pth_path+'/decoder_best.pth'))
```

## Parameters

### Decoder Parameters

- `message_length` (int, default=30): Length of the watermark message
- `decoder_channels` (int, default=64): Number of channels in decoder
- `in_channel` (int, default=3): Number of input channels (RGB)
- `use_moe` (bool, default=False): Enable MOE architecture
- `num_experts` (int, default=4): Number of expert networks per MOE block

### EncoderDecoder Parameters

- `H` (int): Height of input images
- `W` (int): Width of input images
- `message_length` (int): Length of watermark message
- `noise_layers` (list): Noise layer configuration
- `blocks` (int, default=4): Number of encoder blocks
- `use_moe` (bool, default=False): Enable MOE in decoder
- `num_experts` (int, default=4): Number of experts per MOE block

## Benefits

1. **Adaptive Feature Extraction**: Different experts can specialize in different image characteristics
2. **Improved Robustness**: Multiple expert pathways provide redundancy against attacks
3. **Dynamic Routing**: Gating network learns optimal expert combinations per input
4. **Backward Compatible**: Can be disabled to use original architecture

## Performance Considerations

- **Training Time**: Approximately 1.65x slower due to increased parameters
- **Memory Usage**: Requires ~65% more GPU memory
- **Inference Time**: Slightly slower but often worth the improved accuracy
- **Model Size**: Checkpoint files will be approximately 1.65x larger

## Testing

A comprehensive test suite is provided in `test_moe.py`:

```bash
python test_moe.py
```

The test suite verifies:
- Original decoder functionality (backward compatibility)
- MOE decoder output shapes and correctness
- Integration with encoder pipeline
- Parameter count comparison

## Example Output

```
==================================================
Testing MoE Decoder
==================================================
✓ Original decoder test passed!
✓ MoE decoder test passed!
✓ Standalone MoE decoder test passed!

Parameter Count Comparison
   Original Decoder Parameters: 1,897,759
   MoE Decoder Parameters: 3,130,847
   Increase: 1,233,088 (64.98%)
```

## Future Enhancements

Potential improvements for the MOE architecture:

1. **Sparse Gating**: Only activate top-k experts to reduce computation
2. **Load Balancing**: Ensure experts are utilized evenly during training
3. **Expert Specialization**: Add auxiliary losses to encourage expert diversity
4. **Hierarchical Routing**: Multi-level gating for finer-grained control
5. **Dynamic Expert Count**: Adjust number of experts based on input complexity

## References

- Original DWSF Paper: "Practical Deep Dispersed Watermarking with Synchronization and Fusion"
- Mixture of Experts: Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer" (2017)
