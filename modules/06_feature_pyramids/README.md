# Module 06 — Feature Pyramid Networks

## Learning Objectives

1. Understand the multi-scale detection problem and why naive approaches fail.
2. Build an FPN from scratch and explain top-down pathway + lateral connections.
3. Implement PANet's bottom-up pathway addition.
4. Understand BiFPN's weighted bi-directional fusion.
5. Assign objects to pyramid levels using the area-based rule.

---

## The Multi-Scale Problem

Objects in real images appear at vastly different scales.  A 20-pixel pedestrian
and a 500-pixel pedestrian are the same class but look completely different.

**Pre-FPN approaches:**
- Image pyramids: run detector at multiple image scales (slow).
- Single-scale features: only use the last conv layer (loses resolution for small objects).
- SSD: predict from multiple layers independently (no feature sharing).

---

## Feature Pyramid Network — FPN (Lin et al., 2017)

FPN adds two pathways:
1. **Bottom-up pathway** (standard forward pass): {C2, C3, C4, C5} from ResNet.
2. **Top-down pathway**: upsample C5 → P5 → P4 → P3 → P2.
3. **Lateral connections**: 1×1 conv on each Ci, added to upsampled Pi+1.

Result: {P2, P3, P4, P5, P6} — semantically rich at all scales.

**Level assignment:** object with area A is assigned to level P_k where
`k = floor(k0 + log2(sqrt(A) / 224))` with `k0 = 4`.

---

## PANet (Liu et al., 2018)
Adds a **bottom-up path augmentation** on top of FPN: a second pass from P2
back up to P5, so every level has both low-level and high-level context.

## BiFPN (Tan et al., 2020 — EfficientDet)
- Removes nodes with only one input edge.
- Adds **weighted** (learnable) fusion: `O = Σ w_i · I_i / (Σ w_i + ε)`.
- Repeated multiple times (multiple BiFPN layers).

---

## Key References

- Lin et al. (2017) *Feature Pyramid Networks for Object Detection*. CVPR.
- Liu et al. (2018) *Path Aggregation Network for Instance Segmentation*. CVPR.
- Tan et al. (2020) *EfficientDet: Scalable and Efficient Object Detection*. CVPR.
