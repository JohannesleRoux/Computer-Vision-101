# Module 04 — Region Proposals

## Learning Objectives

1. Understand the computational cost of sliding-window detection and why it was the bottleneck in early detectors.
2. Implement image pyramids for scale-invariant detection.
3. Use selective search to generate category-independent region proposals.
4. Implement non-maximum suppression (NMS) to deduplicate detections.
5. Measure proposal quality using recall vs. number-of-proposals curves.

---

## Key Concepts

### Sliding Window
Exhaustive evaluation of a classifier at every position and scale.
Cost: O(W × H × scales × aspect_ratios × classifier_time).
For a 500×500 image with 5 scales and 3 aspect ratios, this can mean millions of windows.

### Image Pyramids
Build a multi-scale stack of downsampled images.  The detector runs at a fixed
size; scaling handles objects at different sizes.  A Gaussian pyramid is constructed
by low-pass filtering then subsampling by factor 2 at each level.

### Selective Search (Uijlings et al., 2013)
Combines over-segmentation (SLIC / Felzenszwalb) with a greedy hierarchical grouping
strategy based on colour, texture, size, and fill similarity.  Generates ~2000 diverse
proposals per image in ~1-2 seconds.

Key properties:
- High recall even at ~1000 proposals
- Proposals at multiple scales by construction
- No learning required

### Non-Maximum Suppression (NMS)
Deduplicates overlapping detections by keeping the highest-scoring box and removing
others with IoU ≥ threshold.

Algorithm:
1. Sort boxes by score (descending).
2. Keep the top box.
3. Remove all boxes with IoU ≥ threshold.
4. Repeat for remaining boxes.

**Soft-NMS** (Bodla et al., 2017) decays scores of overlapping boxes instead of
discarding them — better for crowds.

---

## Key References

- Uijlings et al. (2013) *Selective Search for Object Recognition*. IJCV.
- Felzenszwalb & Huttenlocher (2004) *Efficient Graph-Based Image Segmentation*. IJCV.
- Bodla et al. (2017) *Soft-NMS — Improving Object Detection With One Line of Code*. ICCV.
- Viola & Jones (2001) *Rapid Object Detection using a Boosted Cascade of Simple Features*. CVPR. (Classic sliding window)
