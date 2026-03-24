# Module 09 — YOLO Family

## Learning Objectives

1. Understand YOLOv1's grid-cell formulation and why it was revolutionary.
2. Explain anchor boxes and multi-scale prediction in YOLOv3.
3. Understand anchor-free detection (FCOS, YOLOv8).
4. Use the Ultralytics API for training and inference.

---

## YOLO Evolution

| Version | Year | Key Innovation |
|---------|------|---------------|
| YOLOv1 | 2016 | Grid cells, single forward pass |
| YOLOv2 | 2017 | Anchor boxes, batch norm, multi-scale training |
| YOLOv3 | 2018 | Multi-scale detection, Darknet-53 |
| YOLOv4 | 2020 | Bag of freebies/specials, CSP, PANet |
| YOLOv5 | 2020 | PyTorch, AutoAnchor, Focus layer |
| YOLOX  | 2021 | Anchor-free, decoupled head |
| YOLOv8 | 2023 | Anchor-free, C2f module |
| YOLOv9 | 2024 | GELAN, Programmable Gradient Information |
| YOLOv10| 2024 | NMS-free dual-label assignment |
| YOLOv11| 2024 | Efficient C3k2 modules |

---

## YOLOv1 Formulation

Divides the image into an S×S grid.  Each cell predicts:
- B bounding boxes: (x, y, w, h, confidence)
- C class probabilities

Output tensor: S × S × (B×5 + C)

Loss = λ_coord · Σ box_loss + confidence_loss + class_loss

---

## Anchor-Free Detection

FCOS (Fully Convolutional One-Stage) predicts:
- Distance to box edges (l, r, t, b) from each pixel inside a box.
- Centerness score to suppress low-quality predictions.

YOLOv8 uses the same anchor-free, decoupled-head approach.

---

## Key References

- Redmon et al. (2016) *You Only Look Once: Unified, Real-Time Object Detection*. CVPR.
- Redmon & Farhadi (2018) *YOLOv3: An Incremental Improvement*. arXiv.
- Tian et al. (2019) *FCOS: Fully Convolutional One-Stage Object Detection*. ICCV.
- Jocher et al. (2023) [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
