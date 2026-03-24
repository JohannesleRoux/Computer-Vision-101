"""
utils/datasets.py
-----------------
Dataset helpers for the Computer Vision 101 course.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None  # type: ignore

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None  # type: ignore
    Dataset = object  # type: ignore


# ---------------------------------------------------------------------------
# COCO subset downloader
# ---------------------------------------------------------------------------

def download_coco_subset(
    n_images: int = 100,
    split: str = "val2017",
    output_dir: str = "data/coco",
) -> Path:
    """Download a small subset of COCO images and annotations.

    Downloads *n_images* images from the COCO ``split`` (default: ``val2017``)
    together with the instance annotation JSON.  Uses the public COCO API
    endpoint.

    Parameters
    ----------
    n_images:
        How many images to download.
    split:
        COCO split name (``"val2017"`` or ``"train2017"``).
    output_dir:
        Local directory where images and annotations will be saved.

    Returns
    -------
    Path
        Path to the *output_dir* that now contains ``images/`` and
        ``annotations/``.
    """
    try:
        import requests
    except ImportError as exc:
        raise ImportError("Install `requests` to download COCO data.") from exc

    out = Path(output_dir)
    img_dir = out / "images" / split
    ann_dir = out / "annotations"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    ann_url = (
        f"http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    )
    ann_file = ann_dir / f"instances_{split}.json"

    if not ann_file.exists():
        print(f"Downloading COCO annotations for {split} …")
        # Try a lighter approach: download only the minimal JSON via COCO API
        # Fallback: point at a mirrored single-file annotation endpoint
        try:
            import urllib.request
            url = (
                "https://storage.googleapis.com/coco-dataset/"
                f"annotations/instances_{split}.json"
            )
            urllib.request.urlretrieve(url, str(ann_file))
        except Exception:
            print(
                "Could not auto-download annotations. "
                f"Please place instances_{split}.json in {ann_dir}."
            )
            return out

    with open(ann_file) as f:
        coco = json.load(f)

    images = coco["images"][:n_images]
    print(f"Downloading {len(images)} images …")
    for img_info in images:
        dest = img_dir / img_info["file_name"]
        if dest.exists():
            continue
        url = img_info["coco_url"]
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
        except Exception as exc:
            print(f"  Warning: could not download {url}: {exc}")

    # Write subset annotation file
    subset_ids = {img["id"] for img in images}
    subset_ann = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": coco["categories"],
        "images": images,
        "annotations": [
            a for a in coco["annotations"] if a["image_id"] in subset_ids
        ],
    }
    subset_path = ann_dir / f"instances_{split}_subset{n_images}.json"
    with open(subset_path, "w") as f:
        json.dump(subset_ann, f)

    print(f"Done. Data saved to {out}")
    return out


# ---------------------------------------------------------------------------
# Penn-Fudan Pedestrian Dataset
# ---------------------------------------------------------------------------

class PennFudanDataset(Dataset):
    """Penn-Fudan Pedestrian Dataset for Mask R-CNN tutorials.

    Expects the standard Penn-Fudan layout::

        root/
          PNGImages/   *.png
          PedMasks/    *_mask.png

    Parameters
    ----------
    root:
        Path to the dataset root directory.
    transforms:
        Optional callable applied to ``(image, target)`` pairs.
    """

    def __init__(self, root: str, transforms: Optional[Callable] = None):
        self.root = Path(root)
        self.transforms = transforms
        self.imgs = sorted((self.root / "PNGImages").glob("*.png"))
        self.masks = sorted((self.root / "PedMasks").glob("*.png"))

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int) -> Tuple:
        img_path = self.imgs[idx]
        mask_path = self.masks[idx]

        img = PILImage.open(img_path).convert("RGB")
        mask = PILImage.open(mask_path)
        mask = np.array(mask)

        # Each colour in the mask encodes a different instance.
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids != 0]  # remove background

        masks_bin = (mask == obj_ids[:, None, None]).astype(np.uint8)

        # Derive bounding boxes from masks.
        boxes = []
        valid_masks = []
        for m in masks_bin:
            pos = np.where(m)
            if len(pos[0]) == 0:
                continue
            y1, y2 = int(pos[0].min()), int(pos[0].max())
            x1, x2 = int(pos[1].min()), int(pos[1].max())
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                valid_masks.append(m)

        if torch is not None and len(boxes) > 0:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.ones((len(boxes),), dtype=torch.int64)
            masks_t = torch.as_tensor(np.array(valid_masks), dtype=torch.uint8)
            image_id = torch.tensor([idx])
            area = (boxes_t[:, 3] - boxes_t[:, 1]) * (boxes_t[:, 2] - boxes_t[:, 0])
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
            target = {
                "boxes": boxes_t,
                "labels": labels_t,
                "masks": masks_t,
                "image_id": image_id,
                "area": area,
                "iscrowd": iscrowd,
            }
        else:
            target = {
                "boxes": np.array(boxes, dtype=np.float32),
                "masks": np.array(valid_masks, dtype=np.uint8),
            }

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target


# ---------------------------------------------------------------------------
# STL-10 unlabeled wrapper for self-supervised learning
# ---------------------------------------------------------------------------

class STL10Unlabeled(Dataset):
    """Wrapper around ``torchvision.datasets.STL10`` for self-supervised
    learning experiments in module 12.

    Returns pairs of augmented views of each image when a
    ``pair_transform`` is supplied.

    Parameters
    ----------
    root:
        Dataset download directory.
    split:
        ``"unlabeled"`` (default) or ``"train"`` / ``"test"``.
    pair_transform:
        If provided, the dataset returns ``(view1, view2)`` tuples; both
        views are produced by calling *pair_transform* on the same PIL image.
    transform:
        Standard single-view transform (used when *pair_transform* is None).
    download:
        Download the dataset if not already present.
    """

    def __init__(
        self,
        root: str = "data/stl10",
        split: str = "unlabeled",
        pair_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        download: bool = True,
    ):
        try:
            from torchvision.datasets import STL10
        except ImportError as exc:
            raise ImportError("Install `torchvision` to use STL10Unlabeled.") from exc

        self.base = STL10(root, split=split, transform=None, download=download)
        self.pair_transform = pair_transform
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        img, label = self.base[idx]
        if not isinstance(img, PILImage.Image):
            img = PILImage.fromarray(img)

        if self.pair_transform is not None:
            return self.pair_transform(img), self.pair_transform(img)
        if self.transform is not None:
            return self.transform(img), label
        return img, label
