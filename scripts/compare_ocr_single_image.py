#!/usr/bin/env python3
"""Create one visual comparison image for EasyOCR vs Tesseract on a given input image.

Usage:
  python scripts/compare_ocr_single_image.py --image images/new100.jpg
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime

import cv2
import numpy as np

# Ensure project-root imports work when running this script directly.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from plate_detector import PlateDetector


def _sanitize(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def _pick_best_plate(detections: list[dict]) -> dict | None:
    if not detections:
        return None

    def score(item: dict) -> float:
        return float(item.get("detection_conf", 0.0)) + 0.4 * float(item.get("ocr_conf", 0.0))

    return max(detections, key=score)


def _safe_crop(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(1, min(w, x2))
    y2 = max(1, min(h, y2))
    crop = image[y1:y2, x1:x2]
    return crop


def _draw_text_block(canvas: np.ndarray, lines: list[str], x: int, y: int, color=(30, 30, 30)) -> None:
    dy = 34
    for i, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (x, y + i * dy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            color,
            2,
            cv2.LINE_AA,
        )


def build_comparison(image_path: str, output_path: str, use_gpu: bool = False) -> tuple[str, str]:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    detector = PlateDetector(config.PLATE_MODEL_PATH, use_gpu=use_gpu, use_tesseract=True)
    detections = detector.detect(image)

    best = _pick_best_plate(detections)
    if not best:
        raise RuntimeError("No plate detected in this image.")

    bbox = best["bbox"]
    crop = _safe_crop(image, bbox)
    if crop.size == 0:
        raise RuntimeError("Plate crop is empty after detection.")

    # Match project behavior: super-resolve small crop.
    if crop.shape[0] < 150:
        crop = detector._super_resolve(crop, target_height=150)

    # Preprocess once for both OCR engines.
    variants = detector._preprocess_for_ocr(crop)
    base_for_ocr = variants[0] if variants else crop

    easy_text, easy_conf = detector._ocr_with_easyocr(base_for_ocr)
    tess_text, tess_conf = detector._ocr_with_tesseract(base_for_ocr)

    easy_text = _sanitize(easy_text)
    tess_text = _sanitize(tess_text)

    x1, y1, x2, y2 = bbox
    vis = image.copy()
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 180, 255), 3)
    cv2.putText(
        vis,
        f"Detected Plate (det_conf={best.get('detection_conf', 0.0):.2f})",
        (x1, max(28, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 180, 255),
        2,
        cv2.LINE_AA,
    )

    # Prepare a consistent crop panel.
    panel_h = 280
    panel_w = 520
    crop_show = crop
    ch, cw = crop_show.shape[:2]
    scale = min((panel_w - 40) / max(cw, 1), (panel_h - 80) / max(ch, 1))
    nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))
    crop_resized = cv2.resize(crop_show, (nw, nh), interpolation=cv2.INTER_CUBIC)

    crop_panel = np.full((panel_h, panel_w, 3), 245, dtype=np.uint8)
    cx = (panel_w - nw) // 2
    cy = 40 + (panel_h - 80 - nh) // 2
    crop_panel[cy:cy + nh, cx:cx + nw] = crop_resized if crop_resized.ndim == 3 else cv2.cvtColor(crop_resized, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(crop_panel, (cx - 2, cy - 2), (cx + nw + 2, cy + nh + 2), (60, 60, 60), 2)
    cv2.putText(crop_panel, "Plate Crop Used For OCR", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)

    # Result text panel.
    text_panel = np.full((panel_h, 780, 3), 255, dtype=np.uint8)
    cv2.putText(text_panel, "OCR Comparison Output", (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (10, 10, 10), 2, cv2.LINE_AA)
    lines = [
        f"EasyOCR   : {easy_text or 'NOT_DETECTED'}",
        f"Confidence: {easy_conf:.2f}",
        "",
        f"Tesseract : {tess_text or 'NOT_DETECTED'}",
        f"Confidence: {tess_conf:.2f}",
        "",
        f"Final (pipeline best): {best.get('text', 'NOT_DETECTED')}",
        f"Final OCR method     : {best.get('ocr_method', 'none')}",
    ]
    _draw_text_block(text_panel, lines, x=24, y=86)

    # Upper section: original image with bbox.
    target_w = crop_panel.shape[1] + text_panel.shape[1]
    vh, vw = vis.shape[:2]
    scale_top = min(target_w / max(vw, 1), 540 / max(vh, 1))
    top_w, top_h = int(vw * scale_top), int(vh * scale_top)
    vis_top = cv2.resize(vis, (top_w, top_h), interpolation=cv2.INTER_AREA)
    top_canvas = np.full((top_h + 30, target_w, 3), 250, dtype=np.uint8)
    top_canvas[30:30 + top_h, (target_w - top_w) // 2:(target_w - top_w) // 2 + top_w] = vis_top
    cv2.putText(top_canvas, "Input Image + Plate Detection", (20, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (15, 15, 15), 2, cv2.LINE_AA)

    bottom = np.concatenate([crop_panel, text_panel], axis=1)
    final = np.concatenate([top_canvas, bottom], axis=0)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, final)

    txt_output = os.path.splitext(output_path)[0] + ".txt"
    with open(txt_output, "w", encoding="utf-8") as f:
        f.write(f"Image: {image_path}\n")
        f.write(f"Detection bbox: {bbox}\n")
        f.write(f"Detection confidence: {best.get('detection_conf', 0.0):.4f}\n")
        f.write(f"EasyOCR text: {easy_text or 'NOT_DETECTED'}\n")
        f.write(f"EasyOCR conf: {easy_conf:.4f}\n")
        f.write(f"Tesseract text: {tess_text or 'NOT_DETECTED'}\n")
        f.write(f"Tesseract conf: {tess_conf:.4f}\n")
        f.write(f"Pipeline final text: {best.get('text', 'NOT_DETECTED')}\n")
        f.write(f"Pipeline method: {best.get('ocr_method', 'none')}\n")

    return output_path, txt_output


def main() -> None:
    parser = argparse.ArgumentParser(description="EasyOCR vs Tesseract comparison on one image.")
    parser.add_argument("--image", default="images/new100.jpg", help="Path to input image")
    parser.add_argument("--output", default="outputs/ocr_compare_new100.jpg", help="Output comparison image path")
    parser.add_argument("--use-gpu", action="store_true", help="Enable GPU if available")
    args = parser.parse_args()

    out_img, out_txt = build_comparison(args.image, args.output, use_gpu=args.use_gpu)
    print(f"Saved image: {out_img}")
    print(f"Saved text : {out_txt}")


if __name__ == "__main__":
    main()
