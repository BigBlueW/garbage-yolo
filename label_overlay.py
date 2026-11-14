"""Shared label drawing for test_gui.py and make_demo_gif.py: class badges + grasp-order numbers."""

from __future__ import annotations

import colorsys

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX

# OpenCV uses BGR colors.  The first recommendation is deliberately reserved
# for red; all remaining detections share a stable color with their layer.
FIRST_PICK_COLOR = (0, 0, 255)
LAYER_COLORS = (
    (0, 190, 0),      # layer 1: green
    (0, 150, 255),    # layer 2: orange
    (255, 120, 0),    # layer 3: blue
    (180, 70, 180),   # layer 4: purple
    (180, 170, 0),    # layer 5: cyan/teal
    (80, 80, 210),    # layer 6: coral
)


def grasp_order_color(layer: int, is_first_pick: bool = False) -> tuple[int, int, int]:
    """Return the stable BGR visualization color for a grasp layer."""
    if is_first_pick:
        return FIRST_PICK_COLOR
    if layer < 1:
        return (160, 160, 160)
    if layer <= len(LAYER_COLORS):
        return LAYER_COLORS[layer - 1]
    # Generate additional stable colors instead of repeating an earlier layer.
    hue = (0.18 + (layer - 1) * 0.61803398875) % 1.0
    if hue < 0.06 or hue > 0.94:  # keep the reserved NEXT red unambiguous
        hue = (hue + 0.10) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.82, 0.88)
    return int(blue * 255), int(green * 255), int(red * 255)


def draw_layer_number(
    image: np.ndarray,
    center: tuple[float, float],
    layer: int,
    radius: int,
    is_first_pick: bool = False,
) -> None:
    """Draw a layer badge; the first recommended grasp is always red."""
    cx, cy = int(center[0]), int(center[1])
    fill = grasp_order_color(layer, is_first_pick)
    text_color = (255, 255, 255)
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


def draw_labels(
    image: np.ndarray,
    boxes_list,
    colors: dict,
    layers=None,
    show_conf: bool = True,
    ranks=None,
) -> None:
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
        is_first_pick = ranks is not None and ranks[i] == 1
        if is_first_pick:
            text = f"NEXT {text}"
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
            draw_layer_number(image, (text_x + radius, mid_y), layers[i], radius, is_first_pick)
            text_x += 2 * radius + pad
        text_y = mid_y + th // 2
        color = (grasp_order_color(layers[i], is_first_pick) if layers is not None
                 else colors.get(int(b['cls']), (0, 255, 0)))
        cv2.putText(image, text, (text_x, text_y), FONT, scale, (255, 255, 255), thick + 2 * outline, cv2.LINE_AA)
        cv2.putText(image, text, (text_x, text_y), FONT, scale, color, thick, cv2.LINE_AA)
