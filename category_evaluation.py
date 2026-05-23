#!/usr/bin/env python
"""Category-based detection evaluation for helmet, no-helmet, rider, and plate.

Evaluates a YOLO model against YOLO-format validation labels and reports
precision, recall, F1, and support per class.
"""

from __future__ import annotations

import argparse
import json
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

import config

CLASS_NAMES = {
    0: "with helmet",
    1: "without helmet",
    2: "rider",
    3: "number plate",
}


def xywh_to_xyxy(x_center: float, y_center: float, width: float, height: float, image_width: int, image_height: int):
    half_w = width * image_width / 2.0
    half_h = height * image_height / 2.0
    center_x = x_center * image_width
    center_y = y_center * image_height
    return (
        center_x - half_w,
        center_y - half_h,
        center_x + half_w,
        center_y + half_h,
    )


def box_iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def load_labels(label_path: Path, image_width: int, image_height: int):
    boxes = []
    if not label_path.exists():
        return boxes

    with label_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
            boxes.append(
                {
                    "class_id": class_id,
                    "bbox": xywh_to_xyxy(x_center, y_center, width, height, image_width, image_height),
                }
            )
    return boxes


def load_predictions(model: YOLO, image_path: Path, conf_threshold: float):
    image = cv2.imread(str(image_path))
    if image is None:
        return []

    result = model.predict(image, conf=conf_threshold, verbose=False)[0]
    if result.boxes is None:
        return []

    predictions = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        predictions.append(
            {
                "class_id": class_id,
                "confidence": confidence,
                "bbox": (x1, y1, x2, y2),
            }
        )
    return predictions


def match_boxes(ground_truth, predictions, iou_threshold: float):
    per_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    matched_predictions = set()

    grouped_gt = defaultdict(list)
    grouped_pred = defaultdict(list)

    for item in ground_truth:
        grouped_gt[item["class_id"]].append(item["bbox"])
    for item in predictions:
        grouped_pred[item["class_id"]].append(item)

    all_class_ids = sorted(set(grouped_gt.keys()) | set(grouped_pred.keys()))

    for class_id in all_class_ids:
        gt_boxes = grouped_gt[class_id]
        pred_boxes = grouped_pred[class_id]
        used_gt = set()

        scored_predictions = []
        for pred_index, pred in enumerate(pred_boxes):
            best_iou = 0.0
            best_gt_index = None
            for gt_index, gt_box in enumerate(gt_boxes):
                if gt_index in used_gt:
                    continue
                iou_score = box_iou(pred["bbox"], gt_box)
                if iou_score > best_iou:
                    best_iou = iou_score
                    best_gt_index = gt_index
            scored_predictions.append((pred_index, best_iou, best_gt_index))

        scored_predictions.sort(key=lambda item: item[1], reverse=True)

        for pred_index, best_iou, best_gt_index in scored_predictions:
            if best_iou >= iou_threshold and best_gt_index is not None and best_gt_index not in used_gt:
                per_class[class_id]["tp"] += 1
                used_gt.add(best_gt_index)
                matched_predictions.add((class_id, pred_index))
            else:
                per_class[class_id]["fp"] += 1

        per_class[class_id]["fn"] += max(0, len(gt_boxes) - len(used_gt))

    return per_class


def summarize(per_class_counts):
    rows = []
    total_tp = total_fp = total_fn = 0

    for class_id in sorted(CLASS_NAMES.keys()):
        counts = per_class_counts.get(class_id, {"tp": 0, "fp": 0, "fn": 0})
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        rows.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": tp + fn,
            }
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)) if (micro_precision + micro_recall) else 0.0

    return rows, {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": micro_precision,
        "recall": micro_recall,
        "f1": micro_f1,
    }


def evaluate(model_path: str, image_dir: str, label_dir: str, conf_threshold: float, iou_threshold: float, limit: int | None):
    model = YOLO(model_path)
    image_paths = sorted([
        path for path in Path(image_dir).glob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ])

    if limit is not None:
        image_paths = image_paths[:limit]

    per_class_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    processed = 0
    skipped = 0

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            skipped += 1
            continue

        label_path = Path(label_dir) / f"{image_path.stem}.txt"
        ground_truth = load_labels(label_path, image.shape[1], image.shape[0])
        predictions = load_predictions(model, image_path, conf_threshold)

        image_counts = match_boxes(ground_truth, predictions, iou_threshold)
        for class_id, counts in image_counts.items():
            per_class_counts[class_id]["tp"] += counts["tp"]
            per_class_counts[class_id]["fp"] += counts["fp"]
            per_class_counts[class_id]["fn"] += counts["fn"]

        processed += 1

    rows, micro = summarize(per_class_counts)

    print("\n" + "=" * 80)
    print("CATEGORY-BASED DETECTION EVALUATION")
    print("=" * 80)
    print(f"Model:          {model_path}")
    print(f"Images processed:{processed}")
    print(f"Images skipped:  {skipped}")
    print(f"Confidence conf: {conf_threshold:.2f}")
    print(f"IoU threshold:   {iou_threshold:.2f}")
    print()

    print(f"{'Class':18s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>8s}")
    for row in rows:
        print(
            f"{row['class_name']:18s} {row['tp']:4d} {row['fp']:4d} {row['fn']:4d} "
            f"{row['precision']*100:9.1f}% {row['recall']*100:9.1f}% {row['f1']*100:9.1f}% {row['support']:8d}"
        )

    print("-" * 80)
    print(
        f"Micro average       {micro['tp']:4d} {micro['fp']:4d} {micro['fn']:4d} "
        f"{micro['precision']*100:9.1f}% {micro['recall']*100:9.1f}% {micro['f1']*100:9.1f}%"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Category-based detection evaluation")
    parser.add_argument("--model", default=config.HELMET_MODEL_PATH, help="Path to YOLO model weights")
    parser.add_argument("--images", default=r"bikehelmetnumberplate/val/images", help="Validation image folder")
    parser.add_argument("--labels", default=r"bikehelmetnumberplate/val/labels", help="Validation label folder")
    parser.add_argument("--conf", type=float, default=0.25, help="Prediction confidence threshold")
    parser.add_argument("--iou", type=float, default=0.50, help="IoU threshold for matching")
    parser.add_argument("--output", default=None, help="Write results to <path>.json and <path>_classes.csv if provided")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to evaluate")
    return parser.parse_args()


def main():
    args = parse_args()
    # Run evaluation and capture rows + micro summary
    model = YOLO(args.model)
    # reuse evaluate's logic but call evaluate and then summarize from returned counts
    # We'll call evaluate() logic here to produce per_class_counts
    # For simplicity, call evaluate and capture printed output by reusing functions
    # We'll reimplement minimal evaluate loop here to get data structures
    image_paths = sorted([
        path for path in Path(args.images).glob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ])
    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    per_class_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    processed = 0

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        label_path = Path(args.labels) / f"{image_path.stem}.txt"
        ground_truth = load_labels(label_path, image.shape[1], image.shape[0])
        predictions = load_predictions(model, image_path, args.conf)
        image_counts = match_boxes(ground_truth, predictions, args.iou)
        for class_id, counts in image_counts.items():
            per_class_counts[class_id]["tp"] += counts["tp"]
            per_class_counts[class_id]["fp"] += counts["fp"]
            per_class_counts[class_id]["fn"] += counts["fn"]
        processed += 1

    rows, micro = summarize(per_class_counts)

    # Print as before
    print("\n" + "=" * 80)
    print("CATEGORY-BASED DETECTION EVALUATION")
    print("=" * 80)
    print(f"Model:          {args.model}")
    print(f"Images processed:{processed}")
    print(f"Images skipped:  {0}")
    print(f"Confidence conf: {args.conf:.2f}")
    print(f"IoU threshold:   {args.iou:.2f}")
    print()

    print(f"{'Class':18s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>8s}")
    for row in rows:
        print(
            f"{row['class_name']:18s} {row['tp']:4d} {row['fp']:4d} {row['fn']:4d} "
            f"{row['precision']*100:9.1f}% {row['recall']*100:9.1f}% {row['f1']*100:9.1f}% {row['support']:8d}"
        )

    print("-" * 80)
    print(
        f"Micro average       {micro['tp']:4d} {micro['fp']:4d} {micro['fn']:4d} "
        f"{micro['precision']*100:9.1f}% {micro['recall']*100:9.1f}% {micro['f1']*100:9.1f}%"
    )

    # Write output files if requested
    if args.output:
        out_base = Path(args.output)
        out_base.parent.mkdir(parents=True, exist_ok=True)
        payload = {"rows": rows, "micro": micro, "model": args.model, "processed": processed}
        with out_base.open('w', encoding='utf-8') as jf:
            json.dump(payload, jf, indent=2)
        # write class CSV
        csv_path = out_base.with_name(out_base.stem + '_classes.csv')
        with csv_path.open('w', newline='', encoding='utf-8') as cf:
            writer = csv.writer(cf)
            writer.writerow(['class_id','class_name','tp','fp','fn','precision','recall','f1','support'])
            for r in rows:
                writer.writerow([r['class_id'], r['class_name'], r['tp'], r['fp'], r['fn'], f"{r['precision']:.6f}", f"{r['recall']:.6f}", f"{r['f1']:.6f}", r['support']])


if __name__ == "__main__":
    main()