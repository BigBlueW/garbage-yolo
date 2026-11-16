#!/usr/bin/env python3
"""Depth-refined top-surface grasp point from a YOLO OBB (no training).

複製自主 repo `src/common/robot_utils/robot_utils/surface_grasp.py`（那邊 import 會拉到 ROS，
這裡要能在沒 ROS 的電腦跑 test_gui）。演算法改動以主 repo 為準，改完再同步過來。

純邏輯模組：不依賴 rclpy。輸入 aligned depth（mm）＋相機內參＋OBB 四角。

每張圖（:func:`prepare_scene`）：
1. 整張深度圖反投影成點雲，RANSAC 擬合桌面平面 → 每個像素離桌面多高。

每個框（:func:`refine_grasp`）：
2. 種子＝上表面：框中央一半區域的第 75 百分位高度，±band/2 內、與框中央連通的像素。
   比 ``flat_mm`` 還矮的物體回報 ``flat``（扁平物不夾）。
3. 對種子做 robust PCA 擬合平面 → 物體表面法向量。
4. 沿「物體自己的法向量」保留最上方 ``band_below_mm`` 內、與框中央連通的像素
   （斜面整片保留；桌面、下層物體、旁邊更高的物體被切掉），重新擬合；重複。
5. 夾取點＝表面重心；閉合方向＝表面平面內 PCA 短軸；寬度＝沿短軸的跨度。

:func:`surface_grasp_to_T` 轉成與 ``PoseEstimator(mode='simple')`` 同慣例的
``T_obj2cam``（z 軸朝相機、x 軸＝夾爪閉合方向、由上往下夾），可直接接 pick_place。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np



@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole 相機內參（單位：像素）."""

    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class SceneDepth:
    """一張深度圖的前處理結果（camera optical frame，mm）."""

    points: np.ndarray          # HxWx3
    valid: np.ndarray           # HxW bool
    height: np.ndarray          # HxW，離桌面高度（朝相機為正）
    table_normal: np.ndarray    # 桌面法向量（朝相機）
    intrinsics: CameraIntrinsics


@dataclass(frozen=True)
class SurfaceGrasp:
    """單一物體的上表面夾取結果（camera optical frame，mm）."""

    center_px: tuple[float, float]       # 夾取點投影回影像
    point_cam: np.ndarray                # 表面重心
    normal_cam: np.ndarray               # 表面法向量（朝相機）
    closing_axis_cam: np.ndarray         # 夾爪閉合方向（表面平面內）
    tilt_deg: float                      # 表面法向量與桌面法向量夾角
    planarity: float                     # 1 - l3/l2（1 = 完全平）
    height_mm: float                     # 物體頂部離桌面高度
    width_mm: float                      # 沿閉合方向的跨度
    closing_px: tuple[tuple[float, float], tuple[float, float]]   # 閉合線段兩端（影像）
    contour_px: tuple[tuple[int, int], ...]   # 上表面外輪廓（影像，簡化過的多邊形；給 UI 畫）
    num_points: int


def _project(point: np.ndarray, intr: CameraIntrinsics) -> tuple[float, float]:
    x, y, z = point
    return (intr.fx * x / z + intr.cx, intr.fy * y / z + intr.cy)


def fit_table_plane(points: np.ndarray, valid: np.ndarray, *, iters: int = 100,
                    inlier_mm: float = 6.0, stride: int = 7, ransac_stride: int = 29,
                    seed: int = 0) -> Optional[tuple[np.ndarray, float]]:
    """RANSAC 平面 ``n·p + d = 0``（n 朝相機）；點太少回傳 None.

    RANSAC 只在稀疏取樣（``ransac_stride``）上找桌面，最後用較密的取樣（``stride``）
    最小平方重擬合 —— 桌面佔畫面大半，100 次就夠，整張 1280x720 約 10–20 ms。
    """
    dense = points[valid][::stride]
    pts = dense[::max(1, ransac_stride // stride)]
    if len(pts) < 100:
        return None
    rng = np.random.default_rng(seed)
    best_n, best_d, best_count = None, 0.0, -1
    for _ in range(iters):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-6:
            continue
        n /= norm
        d = -n @ a
        count = int(np.count_nonzero(np.abs(pts @ n + d) < inlier_mm))
        if count > best_count:
            best_n, best_d, best_count = n, d, count
    if best_n is None:
        return None
    inliers = dense[np.abs(dense @ best_n + best_d) < inlier_mm]
    centroid = inliers.mean(axis=0)
    n = np.linalg.svd(inliers - centroid, full_matrices=False)[2][2]
    if n[2] > 0:
        n = -n
    return n, float(-n @ centroid)


def prepare_scene(depth_mm: np.ndarray, intr: CameraIntrinsics) -> Optional[SceneDepth]:
    """深度圖（mm，0 = 無效）→ 點雲＋桌面高度；找不到桌面回傳 None."""
    depth = np.asarray(depth_mm, dtype=np.float64)
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    points = np.dstack(((u - intr.cx) * depth / intr.fx,
                        (v - intr.cy) * depth / intr.fy, depth))
    valid = depth > 0
    plane = fit_table_plane(points, valid)
    if plane is None:
        return None
    n, d = plane
    return SceneDepth(points=points, valid=valid, height=points @ n + d,
                      table_normal=n, intrinsics=intr)


def _fit_surface(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Robust PCA 平面（去掉離中位數最遠 10%）→ 重心、朝相機法向量、平面度."""
    dist = np.linalg.norm(pts - np.median(pts, axis=0), axis=1)
    pts = pts[dist <= np.percentile(dist, 90)]
    centroid = pts.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov(pts - centroid, rowvar=False))   # 由小到大
    normal = vectors[:, 0]
    if normal @ -centroid < 0:
        normal = -normal
    planarity = float(np.clip(1.0 - values[0] / max(values[1], 1e-12), 0.0, 1.0))
    return centroid, normal, planarity


def _poly_mask(shape: tuple[int, int], corners: np.ndarray, scale: float,
               offset: np.ndarray) -> np.ndarray:
    center = corners.mean(axis=0)
    poly = (center + (corners - center) * scale - offset).round().astype(np.int32)
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [poly], 1)
    return mask.astype(bool)


def _blob_over(region: np.ndarray, core: np.ndarray) -> Optional[np.ndarray]:
    """``region`` 中與 ``core`` 重疊最多的連通區."""
    count, labels = cv2.connectedComponents(region.astype(np.uint8), connectivity=8)
    if count <= 1:
        return None
    overlap = np.bincount(labels[core], minlength=count)
    overlap[0] = 0
    if overlap.max() == 0:
        return None
    return labels == int(np.argmax(overlap))


def _outline(surface: np.ndarray, offset: np.ndarray,
             epsilon_px: float = 2.0) -> tuple[tuple[int, int], ...]:
    """表面遮罩 → 最大外輪廓（Douglas-Peucker 簡化，通常 10–40 點）。"""
    contours, _ = cv2.findContours(surface.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return ()
    poly = cv2.approxPolyDP(max(contours, key=cv2.contourArea), epsilon_px, True).reshape(-1, 2)
    return tuple((int(x + offset[0]), int(y + offset[1])) for x, y in poly)


def refine_grasp(scene: SceneDepth, corners_px, *, min_height_mm: float = 8.0,
                 flat_mm: float = 15.0, band_below_mm: float = 25.0,
                 core_scale: float = 0.5, enlarge: float = 1.2, min_points: int = 80,
                 iterations: int = 2) -> tuple[Optional[SurfaceGrasp], str]:
    """回傳 ``(grasp, 'ok')`` 或 ``(None, reason)``；reason ∈ flat / no_depth / few_points."""
    corners = np.asarray(corners_px, dtype=np.float64).reshape(4, 2)
    # 只處理放大框的外接矩形（ROI），每個框幾 ms
    big = corners.mean(axis=0) + (corners - corners.mean(axis=0)) * enlarge
    h, w = scene.valid.shape
    x0, y0 = np.clip(np.floor(big.min(axis=0)).astype(int), 0, [w - 1, h - 1])
    x1, y1 = np.clip(np.ceil(big.max(axis=0)).astype(int) + 1, 1, [w, h])
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None, 'no_depth'
    points = scene.points[y0:y1, x0:x1]
    height = scene.height[y0:y1, x0:x1]
    above = scene.valid[y0:y1, x0:x1] & (height > min_height_mm)
    offset = np.array([x0, y0], dtype=np.float64)
    roi = (y1 - y0, x1 - x0)

    core = _poly_mask(roi, corners, core_scale, offset) & above
    if np.count_nonzero(core) < min_points // 4:
        return None, 'no_depth'
    core_h = height[core]
    ref_h = float(np.percentile(core_h, 75))
    if ref_h < flat_mm:
        return None, 'flat'
    region_box = _poly_mask(roi, corners, enlarge, offset) & above

    # 種子＝上表面（依離桌面高度）。不用整個框中央：直立瓶子的框中央含側壁，會擬合成側面。
    surface = _blob_over(region_box & (np.abs(height - ref_h) < band_below_mm / 2 + 2.5), core)
    if surface is None or np.count_nonzero(surface) < min_points:
        return None, 'few_points'
    centroid, normal, planarity = _fit_surface(points[surface])
    for _ in range(iterations):
        # 沿「物體自己的法向量」量距離：斜面整片保留
        along = (points - centroid) @ normal
        top = float(np.percentile(along[core], 95))
        region = region_box & (along > top - band_below_mm) & (along < top + 10.0)
        surface = _blob_over(region, core)
        if surface is None or np.count_nonzero(surface) < min_points:
            return None, 'few_points'
        centroid, normal, planarity = _fit_surface(points[surface])

    pts = points[surface]
    ref = np.array([0.0, 1.0, 0.0]) if abs(normal[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(normal, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    uv = np.column_stack(((pts - centroid) @ e1, (pts - centroid) @ e2))
    _, vecs = np.linalg.eigh(np.cov(uv, rowvar=False))
    minor = vecs[:, 0]
    lo, hi = np.percentile(uv @ minor, [2, 98])
    axis = minor[0] * e1 + minor[1] * e2
    mid = centroid + axis * (lo + hi) / 2
    half = (hi - lo) / 2
    intr = scene.intrinsics
    tilt = float(np.degrees(np.arccos(np.clip(abs(normal @ scene.table_normal), 0.0, 1.0))))
    return SurfaceGrasp(
        center_px=_project(centroid, intr),
        point_cam=centroid,
        normal_cam=normal,
        closing_axis_cam=axis,
        tilt_deg=tilt,
        planarity=planarity,
        height_mm=float(np.percentile(height[surface], 95)),
        width_mm=float(hi - lo),
        closing_px=(_project(mid - axis * half, intr), _project(mid + axis * half, intr)),
        contour_px=_outline(surface, offset),
        num_points=int(np.count_nonzero(surface)),
    ), 'ok'


def surface_grasp_to_T(grasp: SurfaceGrasp) -> np.ndarray:
    """→ ``T_obj2cam``（公尺），同 ``PoseEstimator`` simple 慣例：z=[0,0,-1]、x＝閉合方向（俯視投影）."""
    x = np.array([grasp.closing_axis_cam[0], grasp.closing_axis_cam[1], 0.0])
    if np.linalg.norm(x) < 1e-6:          # 表面幾乎垂直於影像：退回影像 x 軸
        x = np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    z = np.array([0.0, 0.0, -1.0])
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, np.cross(z, x), z])
    T[:3, 3] = grasp.point_cam / 1000.0
    return T
