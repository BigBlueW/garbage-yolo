"""Shared label drawing for test_gui.py and make_demo_gif.py: class badges + grasp-order numbers."""

from __future__ import annotations

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_layer_number(image: np.ndarray, center: tuple[float, float], layer: int, radius: int) -> None:
    """Filled circle with the grasp layer; layer 1 (graspable now) is bright green, deeper layers gray."""
    cx, cy = int(center[0]), int(center[1])
    fill, text_color = ((0, 220, 0), (0, 0, 0)) if layer == 1 else ((110, 110, 110), (255, 255, 255))
    cv2.circle(image, (cx, cy), radius, fill, -1, cv2.LINE_AA)
    cv2.circle(image, (cx, cy), radius, (255, 255, 255), max(1, radius // 8), cv2.LINE_AA)
    scale = radius / 20.0
    thick = max(1, int(round(scale * 2)))
    label = str(layer)
    (tw, th), _ = cv2.getTextSize(label, FONT, scale, thick)
    cv2.putText(image, label, (cx - tw // 2, cy + th // 2), FONT, scale, text_color, thick, cv2.LINE_AA)


def _overlaps(rect, placed) -> bool:
    x1, y1, x2, y2 = rect
    return any(x1 < px2 and px1 < x2 and y1 < py2 and py1 < y2 for px1, py1, px2, py2 in placed)


def draw_labels(image: np.ndarray, boxes_list, colors: dict, layers=None, show_conf: bool = True) -> None:
    """Draw one badge per box ("(1) plastic 0.91") above its top corner, nudged up/down to avoid other badges."""
    img_h, img_w = image.shape[:2]
    scale = max(0.45, min(0.75, img_w / 1700.0))
    thick = max(1, int(round(scale * 2)))
    outline = max(1, int(round(scale * 2)))
    pad = 2 * outline

    placed = []
    # 由上往下放，位置才穩定（RGB 與深度兩格一致）
    for i in sorted(range(len(boxes_list)), key=lambda k: float(boxes_list[k]['pts'][:, 1].min())):
        b = boxes_list[i]
        text = f"{b['name']} {b['conf']:.2f}" if show_conf else b['name']
        (tw, th), baseline = cv2.getTextSize(text, FONT, scale, thick)
        radius = max(6, int(th * 0.75)) if layers is not None else 0
        w = tw + (2 * radius + pad if radius else 0) + 2 * pad
        h = max(th + baseline, 2 * radius) + 2 * pad

        top_pt = b['pts'][np.argmin(b['pts'][:, 1])]
        x = int(np.clip(top_pt[0] - w / 2, 0, max(0, img_w - w)))
        base_y = int(top_pt[1]) - h
        rect = None
        for step in (0, -1, 1, -2, 2, -3, 3):
            y = int(np.clip(base_y + step * h, 0, max(0, img_h - h)))
            candidate = (x, y, x + w, y + h)
            if not _overlaps(candidate, placed):
                rect = candidate
                break
        if rect is None:
            y = int(np.clip(base_y, 0, max(0, img_h - h)))
            rect = (x, y, x + w, y + h)
        placed.append(rect)

        x1, y1, _, y2 = rect
        mid_y = (y1 + y2) // 2
        text_x = x1 + pad
        if radius:
            draw_layer_number(image, (text_x + radius, mid_y), layers[i], radius)
            text_x += 2 * radius + pad
        text_y = mid_y + th // 2
        color = colors.get(int(b['cls']), (0, 255, 0))
        cv2.putText(image, text, (text_x, text_y), FONT, scale, (255, 255, 255), thick + 2 * outline, cv2.LINE_AA)
        cv2.putText(image, text, (text_x, text_y), FONT, scale, color, thick, cv2.LINE_AA)
