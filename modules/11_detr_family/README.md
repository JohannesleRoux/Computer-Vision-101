# Module 11 — DETR Family

## Learning Objectives

1. Understand set prediction and why it eliminates NMS and anchor design.
2. Implement Hungarian matching for bipartite assignment.
3. Understand DETR's encoder-decoder architecture.
4. Explain Deformable DETR's sparse attention mechanism.
5. Use RT-DETR for real-time detection.

---

## DETR (Carion et al., 2020)

**Key insight:** Object detection as a set prediction problem.
- No anchors, no NMS, no hand-designed components.
- Uses the Hungarian algorithm for one-to-one matching between predictions and GT.

Architecture:
1. CNN backbone (ResNet) → flatten → Transformer encoder.
2. N learned object queries → Transformer decoder → N box + class predictions.
3. Hungarian matching: find permutation σ* minimising total set prediction cost.

### Loss
```
L_Hungarian = Σ_i [ -log p_σ(i)(c_i) + L_box(b_i, b̂_σ(i)) ]
```
where:
- `L_box = λ_iou * GIoU + λ_L1 * L1`
- Unmatched predictions are assigned the `∅` (no-object) class.

---

## Deformable DETR (Zhu et al., 2021)

DETR limitations:
- Slow convergence (~500 epochs vs. ~36 for Faster R-CNN).
- O(N²) self-attention memory cost.

Deformable DETR replaces dense attention with **deformable attention**:
- Each query attends to K=4 sampling points per level (not all spatial locations).
- Multi-scale: attends to 4 feature pyramid levels.
- Converges in 10× fewer epochs.

---

## RT-DETR (Zhao et al., 2023)

First real-time end-to-end detector matching YOLOv8 speed:
- Efficient backbone (RT-DETRv2 uses ResNet/PResNet).
- Hybrid encoder: CNN path + Transformer path.
- Uncertainty-minimal query selection.

---

## Key References

- Carion et al. (2020) *End-to-End Object Detection with Transformers*. ECCV.
- Zhu et al. (2021) *Deformable DETR: Deformable Transformers for End-to-End Object Detection*. ICLR.
- Zhao et al. (2023) *DETRs Beat YOLOs on Real-time Object Detection*. CVPR 2024.
