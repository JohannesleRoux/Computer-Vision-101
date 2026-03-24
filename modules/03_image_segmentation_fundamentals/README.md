# Module 03 — Image Segmentation Fundamentals

## Learning Objectives

1. Distinguish semantic, instance, and panoptic segmentation tasks.
2. Understand the FCN architecture and why it was a breakthrough for pixel-wise prediction.
3. Build a U-Net from scratch and explain skip connections.
4. Implement segmentation loss functions (cross-entropy, Dice, Focal) and understand their trade-offs.
5. Compute IoU and Dice metrics at the pixel level.

---

## Segmentation Taxonomy

| Task | Output | Example |
|------|--------|---------|
| **Semantic** | Class label per pixel | "All cars are red" |
| **Instance** | Unique ID per object | "Car #1, Car #2" |
| **Panoptic** | Semantic + instance combined | Things + Stuff |

---

## Key Architectures

### FCN (Fully Convolutional Network, Long et al., 2015)
First end-to-end trainable dense prediction network.  Replaces FC layers with 1×1 convolutions.
Uses transposed convolutions + skip connections from intermediate encoder layers.
FCN-32s: single 32× upsampling; FCN-16s / FCN-8s add fine skip connections.

### U-Net (Ronneberger et al., 2015)
Encoder-decoder with *skip connections* that concatenate encoder feature maps to decoder.
Designed for biomedical image segmentation with very limited training data.
Key innovation: high-resolution detail is preserved via the skip paths.

Architecture:
```
Encoder: 3x3 Conv → ReLU → 3x3 Conv → ReLU → 2x2 MaxPool (repeated 4x)
Bottleneck: 3x3 Conv × 2
Decoder:  2x2 TransposedConv → Cat(skip) → 3x3 Conv × 2 (repeated 4x)
Output: 1x1 Conv → num_classes
```

---

## Loss Functions

| Loss | Formula | Notes |
|------|---------|-------|
| Cross-entropy | `-Σ y·log(p)` | Standard; unstable for class imbalance |
| Dice | `1 - 2|P∩G|/(|P|+|G|)` | Handles imbalance; smooth differentiable form |
| Focal | `-α(1-p)^γ log(p)` | Down-weights easy negatives (Lin et al., 2017) |
| BCE + Dice | Weighted combination | Common in medical imaging |

---

## Key References

- Long et al. (2015) *Fully Convolutional Networks for Semantic Segmentation*. CVPR.
- Ronneberger et al. (2015) *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI.
- Lin et al. (2017) *Focal Loss for Dense Object Detection*. ICCV. (RetinaNet)
- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)
