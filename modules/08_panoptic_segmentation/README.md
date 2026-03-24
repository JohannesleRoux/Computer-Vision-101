# Module 08 — Panoptic Segmentation

## Learning Objectives

1. Understand the panoptic segmentation task and the things/stuff distinction.
2. Compute Panoptic Quality (PQ), Segmentation Quality (SQ), and Recognition Quality (RQ).
3. Understand Panoptic FPN architecture.
4. Use HuggingFace pipelines for panoptic inference.

---

## Things vs. Stuff

| Category | Definition | Examples |
|----------|-----------|---------|
| **Things** | Countable objects with defined shape | Person, car, dog |
| **Stuff** | Amorphous regions without clear boundaries | Sky, road, grass |

Panoptic segmentation unifies both: each pixel gets (class_id, instance_id).
For stuff, instance_id = 0 (all pixels of a stuff class are one segment).

---

## Panoptic Quality Metric

PQ = SQ × RQ where:
- **SQ** (Segmentation Quality) = average IoU of matched pairs.
- **RQ** (Recognition Quality) = F1 score on segment detection.

```
PQ = Σ IoU(p,g) / (TP + 0.5·FP + 0.5·FN)
```

A prediction is matched to a GT segment if IoU > 0.5.

---

## Panoptic FPN (Kirillov et al., 2019)

Combines:
1. **Instance segmentation** branch (Mask R-CNN-style) for things.
2. **Semantic segmentation** branch (FPN → merged → stuff classes) for stuff.
3. **Merge step**: instance predictions override stuff predictions where they overlap.

---

## Key References

- Kirillov et al. (2019) *Panoptic Segmentation*. CVPR.
- Kirillov et al. (2019) *Panoptic Feature Pyramid Networks*. CVPR.
- [HuggingFace Panoptic Segmentation](https://huggingface.co/tasks/image-segmentation)
