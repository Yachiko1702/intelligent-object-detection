#!/usr/bin/env python3
"""Command‑line camera detector using Ultralytics YOLO models.

This module can be executed as a standalone script or imported by other
programs.  It encapsulates capture, inference, and drawing in a
``CameraDetector`` class and exposes a simple CLI for configuration.

Supported pretrained models include YOLOv5, YOLOv8, YOLOv9, YOLOv10,
YOLO11, and YOLO-World variants.  Run ``--list-models`` to see them all.

Example
-------
$ python camera_detector.py                        # default (yolov8n)
$ python camera_detector.py --model yolov8s.pt     # YOLOv8 small
$ python camera_detector.py --model yolo11n.pt     # YOLO 11 nano
$ python camera_detector.py --model yolov5su.pt    # YOLOv5 small
$ python camera_detector.py --async                # threaded capture
"""

from __future__ import annotations

import argparse
import logging
import queue
import threading
import time
from pathlib import Path
import re
import sys
from typing import Optional

import cv2
from ultralytics import YOLO


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments and perform basic validation."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        type=Path,
        default=Path("yolov8n.pt"),
        help="YOLO model path or pretrained name (run --list-models to see all)",
    )
    p.add_argument(
        "--cam", type=int, default=0, help="OpenCV camera index (>=0)"
    )
    p.add_argument(
        "--conf", type=float, default=0.25, help="minimum confidence threshold"
    )
    p.add_argument(
        "--device",
        default="",
        help="Torch device string, e.g. 'cpu', '0' (GPU index) or blank for auto",
    )
    p.add_argument(
        "--async", dest="async_capture", action="store_true",
        help="enable asynchronous camera capture in background thread",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="optional path to save annotated video",
    )
    p.add_argument(
        "--list-classes", dest="list_classes", action="store_true",
        help="print the model's class names and exit",
    )
    p.add_argument(
        "--list-models", dest="list_models", action="store_true",
        help="print supported pretrained model keys and exit",
    )
    args = p.parse_args()

    # validate inputs
    if args.cam < 0:
        p.error("--cam must be >= 0")
    if args.conf < 0 or args.conf > 1:
        p.error("--conf must be between 0 and 1")
    if args.device and not re.match(r"^(cpu|\d+)$", args.device):
        p.error("--device must be 'cpu' or a GPU index like '0'")
    # The user may specify a file path (which should exist) or a
    # pretrained name such as ``yolov8n``/``yolov8s.pt``.  We allow both
    # formats; the YOLO class will automatically download the weights if
    # the name is recognized.
    # Accepted pretrained name patterns (with or without .pt):
    #   yolov5 n/s/m/l/x (u variants)     yolov8 n/s/m/l/x
    #   yolov9 t/s/m/c/e                  yolov10 n/s/m/b/l/x
    #   yolo11 n/s/m/l/x                  yolov8 *-world / *-worldv2
    #   rtdetr l/x
    _KNOWN_RE = re.compile(
        r"^("
        r"yolov5[nsmlx]u?"
        r"|yolov8[nsmlx]"
        r"|yolov8[nsmlx]-world(v2)?"
        r"|yolov9[tsmce]"
        r"|yolov10[nsmbBlx]"
        r"|yolo11[nsmlx]"
        r"|rtdetr-[lx]"
        r")(\.pt)?$"
    )
    if not args.model.exists():
        if not _KNOWN_RE.match(args.model.name):
            p.error(
                f"model {args.model} does not exist and is not a known "
                f"pretrained name.  Run --list-models to see valid keys."
            )

    return args


class CameraDetector:
    """Encapsulates video capture, YOLOv8 inference, and drawing.

    Parameters
    ----------
    model_path : Path
        Path to a YOLO .pt file or pretrained identifier.
    cam_index : int
        Camera index for ``cv2.VideoCapture``.
    conf_thresh : float
        Minimum confidence threshold for displaying detections.
    device : str | None
        Torch device string passed to ``model.to()``.
    async_capture : bool
        If ``True`` starts a background thread that reads frames.
    output_file : Path | None
        If provided, frames will also be written to a video file.
    """

    def __init__(
        self,
        model_path: Path,
        cam_index: int,
        conf_thresh: float,
        device: Optional[str] = None,
        async_capture: bool = False,
        output_file: Optional[Path] = None,
    ) -> None:
        self.model = YOLO(str(model_path))
        if device:
            try:
                self.model.to(device)
            except Exception as exc:
                logger.warning("could not set device %s: %s", device, exc)
        self.cam_index = cam_index
        self.conf_thresh = conf_thresh
        self.async_capture = async_capture
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"failed to open camera {cam_index}")
        self._frame_q: queue.Queue = queue.Queue(maxsize=2)
        self._grabber: Optional[threading.Thread] = None
        if async_capture:
            self._grabber = threading.Thread(target=self._grab_loop, daemon=True)
            self._grabber.start()
        self.writer = None
        if output_file:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            self.writer = cv2.VideoWriter(str(output_file), fourcc, fps, (w, h))

    def _grab_loop(self) -> None:
        """Background thread method that keeps the most recent frame in a queue."""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            try:
                self._frame_q.put(frame, block=False)
            except queue.Full:
                _ = self._frame_q.get()
                self._frame_q.put(frame, block=False)

    def close(self) -> None:
        """Release resources (camera, writer, windows)"""
        if self._grabber:
            self.cap.release()
            self._grabber.join()
        else:
            self.cap.release()
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()

    def _grab_frame(self) -> any:
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("camera read failed")
        return frame

    def _draw(self, frame, results, fps: float) -> None:
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < self.conf_thresh:
                continue
            x0, y0, x1, y1 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = f"{self.model.names[cls]} {conf:.2f}"
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(
                frame, label, (x0, y0 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("YOLOv8 Object Detection", frame)
        if self.writer:
            self.writer.write(frame)

    def run(self) -> None:
        """Main loop: grab frames, run inference, and display results."""
        try:
            while True:
                frame = (
                    self._frame_q.get(timeout=1)
                    if self.async_capture
                    else self._grab_frame()
                )
                start = time.perf_counter()
                results = self.model(frame, verbose=False)[0]
                fps = 1.0 / (time.perf_counter() - start) if start else 0.0
                self._draw(frame, results, fps)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        except KeyboardInterrupt:
            logger.info("interrupted by user")
        finally:
            self.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    if args.list_classes:
        # load the model and print available names
        model = YOLO(str(args.model))
        for idx, name in model.names.items():
            print(f"{idx}: {name}")
        return

    if args.list_models:
        # All pretrained detection models supported by Ultralytics.
        # Each key can be passed directly to --model and will be auto-downloaded.
        model_info = [
            # ── YOLOv5 (updated "u" re-implementations) ──────────────
            ("yolov5nu",  "YOLOv5  nano   – ultralight, fastest v5"),
            ("yolov5su",  "YOLOv5  small  – good balance for v5"),
            ("yolov5mu",  "YOLOv5  medium – mid-range v5"),
            ("yolov5lu",  "YOLOv5  large  – accurate v5"),
            ("yolov5xu",  "YOLOv5  xlarge – most accurate v5"),
            # ── YOLOv8 ──────────────────────────────────────────────
            ("yolov8n",   "YOLOv8  nano   – fastest, least accurate"),
            ("yolov8s",   "YOLOv8  small  – good speed/accuracy balance"),
            ("yolov8m",   "YOLOv8  medium – slower but more accurate"),
            ("yolov8l",   "YOLOv8  large  – higher accuracy"),
            ("yolov8x",   "YOLOv8  xlarge – highest accuracy v8"),
            # ── YOLOv8-World (open-vocabulary) ─────────────────────
            ("yolov8s-world",   "YOLOv8-World small  – open-vocab detection"),
            ("yolov8m-world",   "YOLOv8-World medium – open-vocab detection"),
            ("yolov8l-world",   "YOLOv8-World large  – open-vocab detection"),
            ("yolov8s-worldv2", "YOLOv8-WorldV2 small  – improved open-vocab"),
            ("yolov8m-worldv2", "YOLOv8-WorldV2 medium – improved open-vocab"),
            ("yolov8l-worldv2", "YOLOv8-WorldV2 large  – improved open-vocab"),
            ("yolov8x-worldv2", "YOLOv8-WorldV2 xlarge – improved open-vocab"),
            # ── YOLOv9 ──────────────────────────────────────────────
            ("yolov9t",   "YOLOv9  tiny   – ultralight v9"),
            ("yolov9s",   "YOLOv9  small  – fast v9"),
            ("yolov9m",   "YOLOv9  medium – balanced v9"),
            ("yolov9c",   "YOLOv9  compact – accurate & efficient"),
            ("yolov9e",   "YOLOv9  extended – most accurate v9"),
            # ── YOLOv10 ─────────────────────────────────────────────
            ("yolov10n",  "YOLOv10 nano   – NMS-free, fastest v10"),
            ("yolov10s",  "YOLOv10 small  – NMS-free v10"),
            ("yolov10m",  "YOLOv10 medium – NMS-free v10"),
            ("yolov10b",  "YOLOv10 balanced – NMS-free v10"),
            ("yolov10l",  "YOLOv10 large  – NMS-free v10"),
            ("yolov10x",  "YOLOv10 xlarge – NMS-free, most accurate v10"),
            # ── YOLO11 ──────────────────────────────────────────────
            ("yolo11n",   "YOLO11  nano   – latest gen, fastest"),
            ("yolo11s",   "YOLO11  small  – latest gen, balanced"),
            ("yolo11m",   "YOLO11  medium – latest gen"),
            ("yolo11l",   "YOLO11  large  – latest gen, accurate"),
            ("yolo11x",   "YOLO11  xlarge – latest gen, most accurate"),
            # ── RT-DETR (transformer-based) ─────────────────────────
            ("rtdetr-l",  "RT-DETR large  – real-time transformer detector"),
            ("rtdetr-x",  "RT-DETR xlarge – most accurate transformer"),
        ]
        for key, desc in model_info:
            print(f"{key:20s}  {desc}")
        return

    detector = CameraDetector(
        model_path=args.model,
        cam_index=args.cam,
        conf_thresh=args.conf,
        device=args.device or None,
        async_capture=args.async_capture,
        output_file=args.output,
    )
    detector.run()


if __name__ == "__main__":
    main()
