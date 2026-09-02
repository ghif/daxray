"""TBX11K dataset discovery, annotation parsers, and sample batching."""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

TBX11K_CLASS_TO_ID = {
    "background": 0,
    "activetuberculosis": 1,
    "active": 1,
    "active tb": 1,
    "obsoletepulmonarytuberculosis": 2,
    "latenttuberculosis": 2,
    "latent": 2,
    "latent tb": 2,
    "obsolete": 2,
    "pulmonarytuberculosis": 1,
}

ID_TO_CLASS_NAME = {
    0: "Background",
    1: "Active TB",
    2: "Latent TB",
}


@dataclass(frozen=True)
class BoundingBox:
    """Bounding box in pixel coordinates [x1, y1, x2, y2]."""

    x1: float
    y1: float
    x2: float
    y2: float
    category_id: int
    category_name: str
    confidence: Optional[float] = None

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    def scale(self, scale_x: float, scale_y: float) -> "BoundingBox":
        return BoundingBox(
            x1=self.x1 * scale_x,
            y1=self.y1 * scale_y,
            x2=self.x2 * scale_x,
            y2=self.y2 * scale_y,
            category_id=self.category_id,
            category_name=self.category_name,
            confidence=self.confidence,
        )


@dataclass
class TBX11KSample:
    """Single sample metadata and annotations in TBX11K."""

    image_id: str
    image_path: Path
    width: int
    height: int
    boxes: list[BoundingBox] = field(default_factory=list)
    image_label: int = 0  # 0: Healthy/Negative, 1: Active TB, 2: Latent TB, 3: Other Sick

    @property
    def has_active_tb(self) -> bool:
        return any(b.category_id == 1 for b in self.boxes)

    @property
    def has_latent_tb(self) -> bool:
        return any(b.category_id == 2 for b in self.boxes)

    @property
    def is_tb_positive(self) -> bool:
        return self.has_active_tb or self.has_latent_tb


def parse_voc_xml(xml_path: Path | str) -> list[BoundingBox]:
    """Parses Pascal VOC XML format annotation for TBX11K.

    Args:
        xml_path: Path to XML file.

    Returns:
        List of BoundingBox objects.
    """
    path = Path(xml_path)
    if not path.exists():
        return []

    tree = ET.parse(path)
    root = tree.getroot()

    boxes = []
    for obj in root.findall("object"):
        name_elem = obj.find("name")
        if name_elem is None or not name_elem.text:
            continue
        raw_name = name_elem.text.strip().lower()
        cat_id = TBX11K_CLASS_TO_ID.get(raw_name, 1)

        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        xmin_el = bndbox.find("xmin")
        ymin_el = bndbox.find("ymin")
        xmax_el = bndbox.find("xmax")
        ymax_el = bndbox.find("ymax")

        if xmin_el is None or ymin_el is None or xmax_el is None or ymax_el is None:
            continue

        x1 = float(xmin_el.text)
        y1 = float(ymin_el.text)
        x2 = float(xmax_el.text)
        y2 = float(ymax_el.text)

        boxes.append(
            BoundingBox(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                category_id=cat_id,
                category_name=ID_TO_CLASS_NAME.get(cat_id, "Unknown"),
            )
        )

    return boxes


def parse_coco_json(json_path: Path | str) -> dict[str, list[BoundingBox]]:
    """Parses COCO format JSON annotation file for TBX11K.

    Args:
        json_path: Path to JSON file.

    Returns:
        Dictionary mapping image relative file name (e.g. 'tb/tb0003.png') to list of BoundingBox.
    """
    path = Path(json_path)
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cat_map = {}
    for cat in data.get("categories", []):
        c_id = cat["id"]
        c_name = cat["name"].lower()
        cat_map[c_id] = TBX11K_CLASS_TO_ID.get(c_name, c_id)

    img_id_to_file = {}
    for img in data.get("images", []):
        img_id_to_file[img["id"]] = img["file_name"]

    result = {}
    for ann in data.get("annotations", []):
        img_id = ann["image_id"]
        file_name = img_id_to_file.get(img_id)
        if not file_name:
            continue

        bbox = ann["bbox"]  # [x, y, w, h]
        x1 = float(bbox[0])
        y1 = float(bbox[1])
        x2 = x1 + float(bbox[2])
        y2 = y1 + float(bbox[3])

        cat_id = cat_map.get(ann.get("category_id", 1), 1)
        box = BoundingBox(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            category_id=cat_id,
            category_name=ID_TO_CLASS_NAME.get(cat_id, "Unknown"),
        )
        result.setdefault(file_name, []).append(box)

    return result


class TBX11KDataset:
    """TBX11K Dataset interface for loading images, ground-truth annotations, and splits."""

    def __init__(
        self,
        root_dir: Path | str = "/Users/mghifary/Work/Code/AI/data/TBX11K",
        split_list: Optional[str] = "TBX11K_val.txt",
    ):
        self.root_dir = Path(root_dir)
        self.imgs_dir = self.root_dir / "imgs"
        self.annotations_xml_dir = self.root_dir / "annotations" / "xml"
        self.annotations_json_dir = self.root_dir / "annotations" / "json"
        self.lists_dir = self.root_dir / "lists"

        self.samples: list[TBX11KSample] = []
        if split_list:
            self.load_split(split_list)

    def load_split(self, list_filename: str) -> None:
        """Loads samples from list file (e.g. 'TBX11K_val.txt')."""
        list_path = self.lists_dir / list_filename
        if not list_path.exists():
            raise FileNotFoundError(f"Split list file not found: {list_path}")

        with open(list_path, "r", encoding="utf-8") as f:
            rel_paths = [line.strip() for line in f if line.strip()]

        self.samples = []
        for rel_p in rel_paths:
            img_path = self.imgs_dir / rel_p
            stem = Path(rel_p).stem

            # Try loading corresponding XML annotation if exists
            xml_path = self.annotations_xml_dir / f"{stem}.xml"
            boxes = parse_voc_xml(xml_path) if xml_path.exists() else []

            # Determine image-level label:
            # 0: Healthy, 1: Active TB, 2: Latent TB, 3: Sick (other pulmonary disease)
            if any(b.category_id == 1 for b in boxes):
                label = 1
            elif any(b.category_id == 2 for b in boxes):
                label = 2
            elif rel_p.startswith("health"):
                label = 0
            elif rel_p.startswith("sick"):
                label = 3
            else:
                label = 0

            # Default dummy dims if not opened yet
            sample = TBX11KSample(
                image_id=stem,
                image_path=img_path,
                width=512,
                height=512,
                boxes=boxes,
                image_label=label,
            )
            self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> TBX11KSample:
        return self.samples[idx]

    def load_image(
        self,
        sample: TBX11KSample,
        target_size: tuple[int, int] = (512, 512),
        normalize: bool = True,
    ) -> tuple[np.ndarray, list[BoundingBox]]:
        """Loads and resizes image, returning normalized array (H, W, 3) and scaled ground truth boxes."""
        if not sample.image_path.exists():
            raise FileNotFoundError(f"Image not found: {sample.image_path}")

        img = Image.open(sample.image_path).convert("RGB")
        orig_w, orig_h = img.size
        sample.width = orig_w
        sample.height = orig_h

        if (orig_w, orig_h) != target_size:
            img_resized = img.resize(target_size, Image.Resampling.BILINEAR)
        else:
            img_resized = img

        img_np = np.array(img_resized, dtype=np.float32) / 255.0

        if normalize:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_np = (img_np - mean) / std

        # Scale ground truth boxes
        scale_x = float(target_size[0]) / float(orig_w)
        scale_y = float(target_size[1]) / float(orig_h)
        scaled_boxes = [b.scale(scale_x, scale_y) for b in sample.boxes]

        return img_np, scaled_boxes
