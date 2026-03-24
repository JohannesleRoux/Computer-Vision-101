# Module 07 — Mask R-CNN & Instance Segmentation

## Learning Objectives

1. Understand the distinction between instance and semantic segmentation.
2. Walk through the Mask R-CNN architecture and explain each head.
3. Implement the mask head (small FCN on RoI-Aligned features).
4. Fine-tune Mask R-CNN on a custom dataset.
5. Visualise predicted masks and compare to ground truth.

---

## Mask R-CNN (He et al., 2017)

Extends Faster R-CNN by adding a **mask head** in parallel with the existing
classification and bounding-box regression heads.

Architecture:
```
Image → Backbone (ResNet-FPN) → RPN → RoI Align
                                         ↓
                                  ┌──────────────┐
                                  │ Classification head │ → class + score
                                  │ Box regression head │ → refined box
                                  │ Mask head           │ → binary mask (H×W)
                                  └──────────────┘
```

### Mask Head
- Takes 14×14 RoI-Aligned features.
- 4 × (Conv 3×3 → ReLU), then 2× TransposedConv → 28×28 mask.
- One binary mask per class — predicts all K masks, selects the one matching predicted class.

### Key Insight: RoI Align
Pixel-accurate spatial alignment is critical for mask quality.
RoI Pooling quantisation (±0.5 pixel) causes misalignment that degrades masks.

### Training
- Mask loss: binary cross-entropy on the GT-class mask only (not all classes).
- Combined loss: L = L_cls + L_box + L_mask.

---

## Key References

- He et al. (2017) *Mask R-CNN*. ICCV.
- [Detectron2 model zoo](https://detectron2.readthedocs.io/en/latest/tutorials/model_zoo.html)
- [torchvision Mask R-CNN tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
