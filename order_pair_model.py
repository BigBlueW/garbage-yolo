"""Lightweight pairwise occlusion model shared by training and inference."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

import cv2
import numpy as np
import torch
from torch import nn

from grasp_sequence import SceneObject


PAIR_CLASSES = ("A_before_B", "B_before_A", "no_relation")
GEOMETRY_SIZE = 25


@dataclass(frozen=True)
class RelationDecision:
    status: str
    kind: str | None
    label: int | None
    confidence: float
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def _box_features(obj: SceneObject, width: int, height: int) -> list[float]:
    rect = cv2.minAreaRect(obj.points.astype(np.float32))
    (cx, cy), (box_w, box_h), angle = rect
    radians = math.radians(angle)
    one_hot = [1.0 if obj.class_id == index else 0.0 for index in range(4)]
    return [
        cx / width, cy / height, box_w / width, box_h / height,
        math.sin(radians), math.cos(radians), *one_hot,
    ]


def _polygon_iou(a: np.ndarray, b: np.ndarray) -> float:
    area_a = abs(float(cv2.contourArea(a.astype(np.float32))))
    area_b = abs(float(cv2.contourArea(b.astype(np.float32))))
    intersection, _ = cv2.intersectConvexConvex(a.astype(np.float32), b.astype(np.float32))
    union = area_a + area_b - float(intersection)
    return float(intersection) / union if union > 1e-6 else 0.0


def pair_geometry(a: SceneObject, b: SceneObject, image_shape: Sequence[int]) -> np.ndarray:
    height, width = int(image_shape[0]), int(image_shape[1])
    ax, ay = a.center
    bx, by = b.center
    dx, dy = (bx - ax) / width, (by - ay) / height
    distance = math.sqrt(dx * dx + dy * dy)
    area_ratio = math.log(max(a.area, 1.0) / max(b.area, 1.0)) / 4.0
    values = _box_features(a, width, height) + _box_features(b, width, height)
    values += [dx, dy, distance, _polygon_iou(a.points, b.points), float(np.clip(area_ratio, -1, 1))]
    return np.asarray(values, dtype=np.float32)


def pair_tensor(
    image: np.ndarray,
    a: SceneObject,
    b: SceneObject,
    image_size: int = 160,
    padding: float = 0.12,
) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = image.shape[:2]
    all_points = np.vstack((a.points, b.points))
    x1, y1 = np.floor(all_points.min(axis=0)).astype(int)
    x2, y2 = np.ceil(all_points.max(axis=0)).astype(int)
    pad = int(max(x2 - x1, y2 - y1) * padding)
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(width, x2 + pad + 1), min(height, y2 + pad + 1)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Pair crop has no area")
    crop = image[y1:y2, x1:x2]
    mask_a = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    mask_b = np.zeros_like(mask_a)
    offset = np.asarray([x1, y1], dtype=np.float32)
    cv2.fillConvexPoly(mask_a, np.rint(a.points - offset).astype(np.int32), 255)
    cv2.fillConvexPoly(mask_b, np.rint(b.points - offset).astype(np.int32), 255)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    mask_a = cv2.resize(mask_a, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    mask_b = cv2.resize(mask_b, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    stacked = np.dstack((rgb, mask_a, mask_b)).astype(np.float32) / 255.0
    visual = torch.from_numpy(stacked.transpose(2, 0, 1)).contiguous()
    geometry = torch.from_numpy(pair_geometry(a, b, image.shape))
    return visual, geometry


def swap_pair_batch(visual: torch.Tensor, geometry: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Swap A/B masks and geometry while keeping the shared RGB crop unchanged."""
    if visual.ndim != 4 or visual.shape[1] != 5:
        raise ValueError(f"Expected visual batch BCHW with 5 channels, got {tuple(visual.shape)}")
    if geometry.ndim != 2 or geometry.shape[1] != GEOMETRY_SIZE:
        raise ValueError(f"Expected geometry batch Bx{GEOMETRY_SIZE}, got {tuple(geometry.shape)}")
    swapped_visual = torch.cat((visual[:, :3], visual[:, 4:5], visual[:, 3:4]), dim=1)
    swapped_geometry = torch.cat((
        geometry[:, 10:20], geometry[:, 0:10],
        -geometry[:, 20:22], geometry[:, 22:24], -geometry[:, 24:25],
    ), dim=1)
    return swapped_visual, swapped_geometry


def swap_pair_targets(target: torch.Tensor) -> torch.Tensor:
    mapping = torch.tensor([1, 0, 2], device=target.device)
    return mapping[target]


def polygon_gap(a: SceneObject, b: SceneObject) -> float:
    """Shortest visible-polygon gap in pixels (zero when polygons intersect)."""
    polygon_a = a.points.astype(np.float32)
    polygon_b = b.points.astype(np.float32)
    intersection, _ = cv2.intersectConvexConvex(polygon_a, polygon_b)
    if intersection > 0:
        return 0.0
    distances = [
        abs(cv2.pointPolygonTest(polygon_b, (float(point[0]), float(point[1])), True))
        for point in polygon_a
    ]
    distances.extend(
        abs(cv2.pointPolygonTest(polygon_a, (float(point[0]), float(point[1])), True))
        for point in polygon_b
    )
    return float(min(distances))


def pair_candidate_kind(
    a: SceneObject,
    b: SceneObject,
    image_shape: Sequence[int],
    near_gap_ratio: float = 0.007,
) -> str | None:
    """Classify a pair as intersecting, near-visible-parts, or unrelated.

    Visible-only OBBs may not intersect even when the physical objects overlap.
    In the local annotations, five of six such positive pairs are within 0.61%
    of the image diagonal.  A 0.7% near tier preserves those cases while
    rejecting the observed false 5->9 edge (2.20%).
    """
    gap = polygon_gap(a, b)
    if gap == 0.0:
        return "intersect"
    height, width = float(image_shape[0]), float(image_shape[1])
    diagonal = max(math.hypot(width, height), 1.0)
    return "near" if gap / diagonal <= near_gap_ratio else None


def pair_is_candidate(a: SceneObject, b: SceneObject, image_shape: Sequence[int], proximity: float = 1.35) -> bool:
    """Compatibility wrapper for dataset building and graph evaluation."""
    return pair_candidate_kind(a, b, image_shape) is not None


@torch.no_grad()
def predict_bidirectional_decision(
    model: nn.Module,
    image: np.ndarray,
    a: SceneObject,
    b: SceneObject,
    image_size: int,
    device: torch.device,
    threshold: float,
    near_threshold: float = 0.90,
) -> RelationDecision:
    """Return relation/no-relation/uncertain with bidirectional evidence."""
    kind = pair_candidate_kind(a, b, image.shape)
    if kind is None:
        return RelationDecision("not_candidate", None, None, 0.0, "visible polygons are too far apart")
    visual_ab, geometry_ab = pair_tensor(image, a, b, image_size)
    visual_ba, geometry_ba = pair_tensor(image, b, a, image_size)
    logits = model(
        torch.stack((visual_ab, visual_ba)).to(device),
        torch.stack((geometry_ab, geometry_ba)).to(device),
    )
    probabilities = torch.softmax(logits, dim=1)
    ab = probabilities[0]
    # Convert the BA prediction back to the original AB class convention.
    ba_as_ab = probabilities[1][torch.tensor([1, 0, 2], device=device)]
    label_ab = int(ab.argmax().item())
    label_ba = int(ba_as_ab.argmax().item())
    required = near_threshold if kind == "near" else threshold
    confidence = min(float(ab[label_ab].item()), float(ba_as_ab[label_ba].item()))
    if label_ab != label_ba:
        return RelationDecision("uncertain", kind, None, confidence, "AB/BA predictions disagree")
    if label_ab == 2:
        return RelationDecision("no_relation", kind, 2, confidence, "both directions predict no relation")
    if confidence < required:
        return RelationDecision("uncertain", kind, label_ab, confidence, f"confidence below {required:.2f}")
    combined = float(((ab[label_ab] + ba_as_ab[label_ab]) / 2).item())
    return RelationDecision("relation", kind, label_ab, combined, "bidirectional agreement")


def predict_bidirectional_relation(*args, **kwargs) -> tuple[int, float] | None:
    """Backward-compatible accepted-relation-only interface."""
    decision = predict_bidirectional_decision(*args, **kwargs)
    if decision.status != "relation" or decision.label is None:
        return None
    return decision.label, decision.confidence


class PairwiseOrderNet(nn.Module):
    def __init__(self, geometry_size: int = GEOMETRY_SIZE) -> None:
        super().__init__()
        self.visual = nn.Sequential(
            nn.Conv2d(5, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.geometry = nn.Sequential(nn.Linear(geometry_size, 48), nn.ReLU(inplace=True), nn.Dropout(0.15))
        self.head = nn.Sequential(nn.Linear(144, 64), nn.ReLU(inplace=True), nn.Dropout(0.25), nn.Linear(64, 3))

    def forward(self, visual: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat((self.visual(visual), self.geometry(geometry)), dim=1))


class MobileNetPairwiseOrderNet(nn.Module):
    """ImageNet RGB backbone + dedicated object-mask and geometry branches."""

    def __init__(self, geometry_size: int = GEOMETRY_SIZE, pretrained: bool = False) -> None:
        super().__init__()
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        self.rgb_backbone = backbone.features
        self.rgb_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.mask_branch = nn.Sequential(
            nn.Conv2d(2, 8, 3, stride=2, padding=1), nn.BatchNorm2d(8), nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.geometry = nn.Sequential(nn.Linear(geometry_size, 64), nn.ReLU(inplace=True), nn.Dropout(0.15))
        self.head = nn.Sequential(
            nn.Linear(576 + 32 + 64, 160), nn.ReLU(inplace=True), nn.Dropout(0.30),
            nn.Linear(160, 3),
        )
        self.register_buffer("rgb_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.rgb_backbone.parameters():
            parameter.requires_grad = trainable

    def forward(self, visual: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        rgb = (visual[:, :3] - self.rgb_mean) / self.rgb_std
        rgb_features = self.rgb_pool(self.rgb_backbone(rgb))
        mask_features = self.mask_branch(visual[:, 3:5])
        geometry_features = self.geometry(geometry)
        return self.head(torch.cat((rgb_features, mask_features, geometry_features), dim=1))


def build_pair_model(
    architecture: str,
    geometry_size: int = GEOMETRY_SIZE,
    pretrained: bool = False,
) -> nn.Module:
    if architecture == "tiny_cnn":
        return PairwiseOrderNet(geometry_size)
    if architecture == "mobilenet_v3_small":
        return MobileNetPairwiseOrderNet(geometry_size, pretrained=pretrained)
    raise ValueError(f"Unknown pair model architecture: {architecture}")


def load_pair_model(path: str, device: torch.device) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    architecture = checkpoint.get("architecture", "tiny_cnn")
    model = build_pair_model(architecture, int(checkpoint.get("geometry_size", GEOMETRY_SIZE)), pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint
