# Computer Vision 101

A comprehensive, hands-on course covering the full spectrum of modern computer vision — from classical image processing to diffusion models. Every concept is grounded in working PyTorch code and interactive Jupyter notebooks.

---

## Table of Modules

| # | Module | Key Topics |
|---|--------|------------|
| 00 | [Image Processing Fundamentals](modules/00_image_processing/) | Pixels, color spaces, convolution, filters, morphology, Fourier transform |
| 01 | [Classic CNN Architectures](modules/01_classic_architectures/) | LeNet, AlexNet, VGG, ResNet — the architectural lineage |
| 02 | [CNNs Deep Dive](modules/02_cnns_deep_dive/) | Receptive fields, depthwise separable, dilated, transposed, group convolutions |
| 03 | [Image Segmentation Fundamentals](modules/03_image_segmentation_fundamentals/) | Semantic / instance / panoptic taxonomy, FCN, U-Net, loss functions |
| 04 | [Region Proposals](modules/04_region_proposals/) | Sliding window, image pyramids, selective search, NMS |
| 05 | [R-CNN Family](modules/05_rcnn_family/) | R-CNN → Fast → Faster R-CNN, RoI Pooling, RoI Align, mAP |
| 06 | [Feature Pyramid Networks](modules/06_feature_pyramids/) | FPN, PANet, BiFPN — multi-scale detection backbones |
| 07 | [Mask R-CNN & Instance Segmentation](modules/07_mask_rcnn_instance_seg/) | Mask head, panoptic extension, fine-tuning on custom data |
| 08 | [Panoptic Segmentation](modules/08_panoptic_segmentation/) | Things vs stuff, Panoptic Quality metric, Panoptic FPN |
| 09 | [YOLO Family](modules/09_yolo_family/) | YOLOv1–v11, anchor-free detection, FCOS, Ultralytics API |
| 10 | [Vision Transformers](modules/10_vision_transformers/) | Self-attention, ViT, Swin Transformer, DeiT |
| 11 | [DETR Family](modules/11_detr_family/) | Set prediction, Hungarian matching, DETR, Deformable DETR, RT-DETR |
| 12 | [Self-Supervised Learning](modules/12_self_supervised_learning/) | SimCLR, MoCo, DINO, MAE — learning without labels |
| 13 | [Foundation Models](modules/13_foundation_models/) | CLIP, SAM / SAM2, Grounding DINO, Florence-2 |
| 14 | [Depth Estimation](modules/14_depth_estimation/) | Monocular depth, MiDaS, Depth Anything v2, point clouds |
| 15 | [3D Vision](modules/15_3d_vision/) | NeRF, 3D Gaussian Splatting, point cloud processing |
| 16 | [Video & Optical Flow](modules/16_video_and_optical_flow/) | Temporal modeling, Lucas-Kanade, RAFT, 3D convolutions, video transformers |
| 17 | [Diffusion Models](modules/17_diffusion_models/) | DDPM, DDIM, latent diffusion, Stable Diffusion, ControlNet |

---

## Prerequisites

| Topic | Resources |
|-------|-----------|
| Python 3.10+ | Basic familiarity with numpy and matplotlib |
| Linear algebra | Vectors, matrices, dot products, eigenvalues |
| Calculus | Gradients, chain rule (for backprop modules) |
| Probability | Distributions, expectations, KL divergence (for SSL/diffusion modules) |
| PyTorch basics | Tensors, autograd — covered lightly in module 01 |

No prior deep learning experience is required, but familiarity with Python and numpy will help.

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/your-org/Computer-Vision-101.git
cd Computer-Vision-101
```

### 2. Run the setup script

```bash
bash setup.sh
```

This creates a `.venv` virtual environment and installs all dependencies.

### 3. Activate and launch

```bash
source .venv/bin/activate
jupyter lab
```

Navigate to any `modules/XX_*/notebook.ipynb` to begin.

### Manual installation

If you prefer to manage the environment yourself:

```bash
pip install -r requirements.txt
pip install -e ".[dev]"
```

### GPU acceleration

The notebooks detect GPU availability automatically. Most exercises run in reasonable time on CPU, but modules 10–17 benefit significantly from a CUDA GPU. Google Colab or Kaggle notebooks (free tier) work well.

---

## Repository Layout

```
Computer-Vision-101/
├── README.md                    ← You are here
├── requirements.txt             ← Python dependencies
├── pyproject.toml               ← Package metadata
├── setup.sh                     ← One-command environment setup
│
├── utils/                       ← Shared utility package (cv101)
│   ├── __init__.py
│   ├── visualization.py         ← Image display, bounding boxes, masks, flow
│   ├── metrics.py               ← IoU, Dice, mAP, PQ, depth metrics
│   ├── datasets.py              ← COCO subset downloader, PennFudan, STL-10
│   └── transforms.py            ← Train/val augmentations, SimCLR, detection
│
├── modules/
│   ├── 00_image_processing/
│   │   ├── README.md
│   │   ├── notebook.ipynb       ← Main notebook with exercises
│   │   ├── notebook_solutions.ipynb
│   │   └── utils.py
│   ├── 01_classic_architectures/
│   │   ├── README.md
│   │   ├── notebook.ipynb
│   │   ├── notebook_solutions.ipynb
│   │   └── models.py
│   └── ...                      ← Same pattern for modules 02–17
│
├── capstone/
│   ├── README.md
│   ├── project_1_object_detection_pipeline.ipynb
│   ├── project_2_segmentation_system.ipynb
│   └── project_3_multimodal_grounding.ipynb
│
└── assets/
    └── images/                  ← Sample images used across notebooks
```

---

## Hardware Notes

| Modules | Minimum | Recommended |
|---------|---------|-------------|
| 00–04 | CPU | CPU |
| 05–09 | CPU (slow) | GPU (4 GB VRAM) |
| 10–13 | GPU (4 GB) | GPU (8–16 GB) |
| 14–17 | GPU (8 GB) | GPU (16–24 GB) |

For the diffusion models module (17), running Stable Diffusion inference requires at least 6 GB VRAM. The notebook includes CPU-fallback paths for all exercises.

**Cloud options:**
- Google Colab (free T4) — suitable for all modules
- Kaggle Notebooks (free P100/T4) — suitable for all modules
- Lambda Labs / Vast.ai — if you need a persistent A100 environment

---

## Shared Utilities

The `utils/` package is installed as `cv101` (editable install via `pip install -e .`). Import it in any notebook:

```python
from utils import show_image, draw_boxes, iou_score
```

See `utils/__init__.py` for the full public API.

---

## Contributing

1. Fork the repository and create a feature branch.
2. Run `pre-commit install` after cloning to enable notebook cell output stripping.
3. Write clear docstrings and include type hints for all public functions.
4. Test notebooks end-to-end before opening a pull request.
5. Follow the existing code style (PEP 8, 88-char line limit).

Bug reports and exercise improvements are especially welcome!

---

## License

This course material is released under the **MIT License**. See `LICENSE` for details.

Model weights downloaded during the notebooks (e.g., from HuggingFace Hub) are subject to their own licenses — see each model card for details.

---

## Citation

If you use this material in a course or research project, please cite:

```
@misc{cv101-2024,
  title  = {Computer Vision 101},
  year   = {2024},
  url    = {https://github.com/your-org/Computer-Vision-101}
}
```
