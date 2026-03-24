# Module 05 — R-CNN Family

## Learning Objectives

1. Trace the evolution from R-CNN to Fast R-CNN to Faster R-CNN.
2. Understand SPP (Spatial Pyramid Pooling) and how it overcomes the fixed-input constraint.
3. Implement RoI Pooling and RoI Align from scratch.
4. Explain the Region Proposal Network (RPN) and its anchor system.
5. Compute mAP for an object detection model.

---

## Architecture Evolution

### R-CNN (Girshick et al., 2014)
1. Generate ~2000 proposals with Selective Search.
2. Warp each to 227×227, extract CNN features (AlexNet).
3. SVM classifiers + bounding-box regressors.

**Bottleneck:** CNN run separately on each proposal (~47s/image).

### SPP-Net (He et al., 2014)
Key insight: run CNN **once** on the full image, then pool proposal features
from the shared feature map using Spatial Pyramid Pooling.
Speed: 24–102× faster than R-CNN.

### Fast R-CNN (Girshick, 2015)
- Single forward pass through CNN on full image.
- **RoI Pooling**: extract fixed-size features for each proposal from feature map.
- Joint classification + regression (single network, end-to-end trainable).
- Bottleneck: selective search (~2s/image).

### Faster R-CNN (Ren et al., 2015)
Replaces selective search with a **Region Proposal Network (RPN)**:
- Shares conv features with detection head.
- Predicts objectness scores and box deltas for anchor boxes at each location.
- 5–17 fps (vs. ~0.5 fps for Fast R-CNN).

---

## Key Components

### Anchor Boxes
Pre-defined reference boxes of multiple scales and aspect ratios placed at
every feature map location.  The RPN predicts offsets (Δx, Δy, Δw, Δh) from
these anchors.

### RoI Pooling vs. RoI Align
- **RoI Pooling**: quantises proposal coordinates to feature map grid → misalignment.
- **RoI Align** (He et al., 2017, Mask R-CNN): bilinear interpolation at exact positions → no quantisation loss, critical for segmentation.

---

## Key References

- Girshick et al. (2014) *Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation*. CVPR.
- He et al. (2014) *Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition*. ECCV.
- Girshick (2015) *Fast R-CNN*. ICCV.
- Ren et al. (2015) *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks*. NeurIPS.
