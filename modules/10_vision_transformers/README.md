# Module 10 — Vision Transformers

## Learning Objectives

1. Derive self-attention from first principles and implement multi-head attention.
2. Understand patch embedding and positional encoding in ViT.
3. Build a minimal ViT encoder in PyTorch.
4. Understand Swin Transformer's shifted windows and hierarchical design.
5. Compare ViT, DeiT, and Swin in terms of accuracy/efficiency trade-offs.

---

## Self-Attention

Given query Q, key K, value V matrices:

```
Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) · V
```

- **Scale** by √d_k to prevent softmax saturation.
- **Multi-head**: run h attention heads in parallel with different projections,
  then concatenate and project: `MHA(Q,K,V) = concat(head_1,...,head_h) W_O`.

---

## Vision Transformer (Dosovitskiy et al., 2020)

1. **Patch embedding**: split image into P×P patches (e.g., 16×16), flatten and linearly project each patch to D dimensions.
2. **CLS token**: prepend a learnable classification token.
3. **Positional embedding**: add learned 1-D positional embeddings.
4. **Transformer encoder**: L layers of MHA + MLP (with LayerNorm and residual connections).
5. **Head**: use CLS token for classification.

### Data efficiency
ViT needs large datasets (JFT-300M) to beat CNNs from scratch.
DeiT (Touvron et al., 2021) adds **knowledge distillation** from a CNN teacher to match ViT-L/16 accuracy on ImageNet-21k with just ImageNet-1k.

---

## Swin Transformer (Liu et al., 2021)

Key innovations:
1. **Hierarchical** feature maps (like ResNet): 4 stages, spatial resolution halves at each.
2. **Window-based** self-attention: compute attention only within local W×W windows → O(n) complexity.
3. **Shifted windows**: alternate between regular and shifted window partitions for cross-window connections.

---

## Key References

- Dosovitskiy et al. (2020) *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale*. ICLR 2021.
- Touvron et al. (2021) *Training Data-Efficient Image Transformers & Distillation through Attention*. ICML.
- Liu et al. (2021) *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows*. ICCV.
- [timm model library](https://github.com/huggingface/pytorch-image-models)
