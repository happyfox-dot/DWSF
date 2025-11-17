# Architecture Comparison: Original vs MOE Decoder

## Original Decoder Architecture

```
Input Image (B, 3, 128, 128)
    ↓
ConvBNRelu (3 → 64 channels)
    ↓
SENet Block (4 layers)
    ↓
SENet_decoder (64 → 128 channels)
    ↓
SENet Block (4 layers)
    ↓
SENet_decoder (64 → 128 channels)
    ↓
SENet Block (4 layers)
    ↓
SENet_decoder (64 → 128 channels)
    ↓
SENet Block (4 layers)
    ↓
SENet_decoder (64 → 128 channels)
    ↓
Conv2d (128 → 1 channel)
    ↓
Flatten & Linear (64 → 30)
    ↓
ReLU Activation
    ↓
Output Message (B, 30)
```

**Parameters: 1,897,759**

---

## MOE Decoder Architecture

```
Input Image (B, 3, 128, 128)
    ↓
ConvBNRelu (3 → 64 channels)
    ↓
╔══════════════════════════════════════╗
║ MoEBlock (4 experts)                  ║
║   ┌─────────┐ ┌─────────┐            ║
║   │Expert 1 │ │Expert 2 │            ║
║   └─────────┘ └─────────┘            ║
║   ┌─────────┐ ┌─────────┐            ║
║   │Expert 3 │ │Expert 4 │            ║
║   └─────────┘ └─────────┘            ║
║          ↓                            ║
║   Gating Network                      ║
║   (adaptive routing)                  ║
║          ↓                            ║
║   Weighted Combination                ║
╚══════════════════════════════════════╝
    ↓
SENet Block (2 layers)
    ↓
SENet_decoder (64 → 128 channels)
    ↓
╔══════════════════════════════════════╗
║ MoEBlock (4 experts)                  ║
║   (adaptive feature extraction)       ║
╚══════════════════════════════════════╝
    ↓
SENet Block (2 layers)
    ↓
SENet_decoder (64 → 128 channels)
    ↓
╔══════════════════════════════════════╗
║ MoEBlock (4 experts)                  ║
║   (adaptive feature extraction)       ║
╚══════════════════════════════════════╝
    ↓
SENet Block (2 layers)
    ↓
SENet_decoder (64 → 128 channels)
    ↓
╔══════════════════════════════════════╗
║ MoEBlock (4 experts)                  ║
║   (adaptive feature extraction)       ║
╚══════════════════════════════════════╝
    ↓
SENet Block (2 layers)
    ↓
SENet_decoder (64 → 128 channels)
    ↓
Conv2d (128 → 1 channel)
    ↓
Flatten & Linear (64 → 30)
    ↓
ReLU Activation
    ↓
Output Message (B, 30)
```

**Parameters: 3,130,847** (+65%)

---

## Key Differences

### 1. Expert Networks
- **Original**: Single path through each SENet block
- **MOE**: 4 parallel expert networks per MOE block

### 2. Adaptive Routing
- **Original**: Fixed computational path
- **MOE**: Gating network learns optimal routing per input

### 3. Feature Extraction
- **Original**: 4 SENet blocks with 4 layers each
- **MOE**: 4 MOE blocks + 2-layer SENet blocks (more specialized)

### 4. Capacity
- **Original**: ~1.9M parameters
- **MOE**: ~3.1M parameters (65% increase)

---

## MOE Block Detail

```
Input Features (B, C, H, W)
    ↓
    ├─────────────────┬─────────────────┬─────────────────┬─────────────────┐
    ↓                 ↓                 ↓                 ↓                 ↓
┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌────────────┐
│ Expert 1  │   │ Expert 2  │   │ Expert 3  │   │ Expert 4  │   │  Gating    │
│ Conv+BN+  │   │ Conv+BN+  │   │ Conv+BN+  │   │ Conv+BN+  │   │  Network   │
│ ReLU x 2  │   │ ReLU x 2  │   │ ReLU x 2  │   │ ReLU x 2  │   │(GlobalPool)│
└───────────┘   └───────────┘   └───────────┘   └───────────┘   │  + FC +   │
    ↓                 ↓                 ↓                 ↓       │  Softmax  │
    │                 │                 │                 │       └────────────┘
    │                 │                 │                 │             ↓
    │                 │                 │                 │       weights (B, 4)
    └─────────────────┴─────────────────┴─────────────────┴─────────────┐
                                    ↓                                     ↓
                          Weighted Combination                            
                         (w1*E1 + w2*E2 + w3*E3 + w4*E4)
                                    ↓
                          Output Features (B, C, H, W)
```

### Gating Network Details
1. **Global Average Pooling**: Captures global context
2. **FC Layer 1**: C → C/4 (dimension reduction)
3. **ReLU**: Non-linearity
4. **FC Layer 2**: C/4 → 4 (one weight per expert)
5. **Softmax**: Normalize weights to sum to 1

---

## Benefits of MOE Architecture

### 1. Adaptive Processing
- Different experts can specialize in different image characteristics
- Watermarks in smooth vs. textured regions handled differently
- Gating learns optimal routing during training

### 2. Improved Robustness
- Multiple expert pathways provide redundancy
- If one expert fails on certain attacks, others compensate
- Better generalization to unseen distortions

### 3. Increased Capacity
- 65% more parameters for learning complex patterns
- Each expert can focus on subset of feature space
- More expressive representations

### 4. Backward Compatible
- Can be disabled (use_moe=False) for original behavior
- Existing trained models still work
- Gradual migration path

---

## Training Considerations

### Memory Requirements
- **Original**: ~X GB GPU memory
- **MOE**: ~1.65X GB GPU memory

### Training Time
- **Original**: ~Y hours/epoch
- **MOE**: ~1.65Y hours/epoch

### Convergence
- MOE may require slightly more epochs to converge
- But often achieves better final performance
- Learning rate may need tuning

### Best Practices
1. Start with pre-trained encoder
2. Use same learning rate schedule as original
3. Monitor gating weights to ensure all experts are used
4. Consider load balancing loss if experts are imbalanced

---

## Performance Expectations

Based on MOE literature and our architecture:

### Expected Improvements
- **Bit Error Rate**: 5-15% relative improvement
- **Robustness**: Better performance on combined attacks
- **Generalization**: Better on unseen image types

### Trade-offs
- **Speed**: ~40% slower inference
- **Memory**: ~65% more GPU memory
- **Model Size**: ~65% larger checkpoint files

---

## When to Use MOE

### Use MOE When:
✅ You need maximum watermark extraction accuracy  
✅ You have sufficient GPU memory (8GB+)  
✅ Training time is not a critical constraint  
✅ Robustness to diverse attacks is important  

### Use Original When:
✅ Fast inference is critical  
✅ Memory is limited (< 4GB)  
✅ Model size needs to be minimal  
✅ Current performance is sufficient  

---

## Migration Path

### Phase 1: Testing
```python
# Test MOE with small dataset
decoder_moe = Decoder(use_moe=True, num_experts=4)
```

### Phase 2: Validation
```python
# Compare on validation set
results_original = evaluate(decoder_original)
results_moe = evaluate(decoder_moe)
```

### Phase 3: Production
```python
# Deploy if improvements are significant
if results_moe.bit_error < results_original.bit_error * 0.90:
    deploy(decoder_moe)
```
