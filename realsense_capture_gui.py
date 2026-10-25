"""Live RGB preview and manual capture for an Intel RealSense camera."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


WINDOW_NAME = "RealSense RGB Capture | SPACE:Save  Q/Esc:Quit"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default="941322070906", help="RealSense serial number")
    parser.add_argument("--output-dir", default="capture_inbox")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


def choose_device(context: rs.context, requested_serial: str):
    devices = list(context.query_devices())
    matches = []
    for device in devices:
        serial = device.get_info(rs.camera_info.serial_number)
        if not requested_serial or serial == requested_serial:
            matches.append(device)
    if not matches:
        found = [device.get_info(rs.camera_info.serial_number) for device in devices]
        raise RuntimeError(f"RealSense {requested_serial!r} not found; connected={found}")
    if len(matches) != 1:
        raise RuntimeError("Multiple RealSense cameras found; pass --serial explicitly")
    return matches[0]


def draw_status(frame: np.ndarray, message: str) -> np.ndarray:
    canvas = frame.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 48), (25, 25, 25), -1)
    cv2.putText(
        canvas,
        message,
        (14, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def save_capture(output_dir: Path, frame: np.ndarray, serial: str) -> Path:
    now = datetime.now()
    stem = f"realsense_{now:%Y%m%d_%H%M%S_%f}"
    image_path = output_dir / f"{stem}.jpg"
    metadata_path = output_dir / f"{stem}.capture.json"
    if not cv2.imwrite(str(image_path), frame):
        raise OSError(f"Failed to save {image_path}")
    metadata = {
        "image": image_path.name,
        "captured_at": now.isoformat(timespec="milliseconds"),
        "camera": "Intel RealSense D435 RGB",
        "serial": serial,
        "width": int(frame.shape[1]),
        "height": int(frame.shape[0]),
        "status": "unlabeled_inbox",
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return image_path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    context = rs.context()
    device = choose_device(context, args.serial)
    serial = device.get_info(rs.camera_info.serial_number)
    usb = (device.get_info(rs.camera_info.usb_type_descriptor)
           if device.supports(rs.camera_info.usb_type_descriptor) else "unknown")
    print(f"camera=RealSense D435 serial={serial} usb={usb}")
    if not usb.startswith("3"):
        print(f"WARNING: USB {usb}; use a USB 3.x port for stable RGB capture")

    pipeline = rs.pipeline(context)
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    pipeline.start(config)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, min(args.width, 1280), min(args.height, 720))
    saved_message = "SPACE: Save photo    Q/Esc: Quit"
    try:
        while True:
            frames = pipeline.wait_for_frames(5000)
            color = frames.get_color_frame()
            if not color:
                continue
            frame = np.asanyarray(color.get_data())
            cv2.imshow(WINDOW_NAME, draw_status(frame, saved_message))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key == 32:
                image_path = save_capture(output_dir, frame, serial)
                saved_message = f"Saved: {image_path.name}"
                print(f"saved={image_path.resolve()}")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
