#!/usr/bin/env python
"""Run full evaluation: detection (category), plate OCR, and pipeline metrics.

Produces JSON and CSV outputs with detailed per-image and aggregated metrics.
"""
from __future__ import annotations

import argparse
import json
import csv
from datetime import datetime
from pathlib import Path
import sys

import cv2
import numpy as np
from ultralytics import YOLO

import config
from category_evaluation import evaluate as category_evaluate, summarize

def run_detection_eval(model_path, images_dir, labels_dir, conf, iou, limit=None):
    # Reuse category_evaluation.evaluate but capture printed output by calling evaluate()
    # category_evaluation.evaluate prints results; instead we'll call its internal evaluate
    from category_evaluation import evaluate as _eval
    # The evaluate function prints; to get numbers, we'll reimplement minimal loop here
    model = YOLO(model_path)
    image_paths = sorted([p for p in Path(images_dir).glob('*') if p.suffix.lower() in {'.jpg','.png','.jpeg'}])
    if limit:
        image_paths = image_paths[:limit]

    per_class_counts = { }
    per_image_results = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        # load labels
        label_path = Path(labels_dir) / f"{image_path.stem}.txt"
        # use category_evaluation.load_labels
        from category_evaluation import load_labels, load_predictions, match_boxes
        gt = load_labels(label_path, image.shape[1], image.shape[0])
        preds = load_predictions(model, image_path, conf)
        counts = match_boxes(gt, preds, iou)
        per_image_results.append({
            'image': str(image_path),
            'gt_count': sum(len(v) for v in [{'a':1}]),
            'counts': counts,
        })
        # aggregate
        for cid, c in counts.items():
            if cid not in per_class_counts:
                per_class_counts[cid] = {'tp':0,'fp':0,'fn':0}
            per_class_counts[cid]['tp'] += c['tp']
            per_class_counts[cid]['fp'] += c['fp']
            per_class_counts[cid]['fn'] += c['fn']

    # summarize using category_evaluation.summarize expects per_class_counts keyed 0..n
    from collections import defaultdict
    full = defaultdict(lambda: {'tp':0,'fp':0,'fn':0})
    for k,v in per_class_counts.items():
        full[k] = v

    rows, micro = summarize(full)
    return {'rows': rows, 'micro': micro, 'per_image': per_image_results}


def run_ocr_eval(plate_detector, images_dir, limit=None):
    # For each image, extract plate crops via labels and run plate_detector._ocr_crop
    results = []
    image_paths = sorted([p for p in Path(images_dir).glob('*') if p.suffix.lower() in {'.jpg','.png','.jpeg'}])
    if limit:
        image_paths = image_paths[:limit]

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        # attempt to find plate detections via plate_detector.detect(image)
        try:
            dets = plate_detector.detect(image)
        except Exception as e:
            dets = []
        # dets format assumed: list of dicts with 'bbox' and 'crop'
        for det in dets:
            crop = det.get('crop')
            ocr = plate_detector._ocr_crop(crop)
            results.append({
                'image': str(image_path),
                'ocr_text': ocr[0] if isinstance(ocr, (list,tuple)) else str(ocr),
                'ocr_conf': ocr[1] if isinstance(ocr, (list,tuple)) and len(ocr)>1 else None,
                'method': ocr[2] if isinstance(ocr, (list,tuple)) and len(ocr)>2 else None,
            })

    # compute simple OCR stats: count non-empty
    total = len(results)
    non_empty = sum(1 for r in results if r['ocr_text'])
    return {'total_crops': total, 'recognized': non_empty, 'details': results}


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_csv(details, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not details:
        return
    keys = sorted(details[0].keys())
    with path.open('w', newline='', encoding='utf-8') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=keys)
        writer.writeheader()
        for r in details:
            writer.writerow(r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--outdir', default='outputs/metrics')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')

    print('Running category evaluation...')
    cat = run_detection_eval(config.HELMET_MODEL_PATH, 'bikehelmetnumberplate/val/images', 'bikehelmetnumberplate/val/labels', 0.25, 0.50, args.limit)
    save_json(cat, outdir / f'category_eval_{timestamp}.json')

    # Create combined per-image CSV summary: image, tp,fp,fn,precision,recall,f1,ocr_crops,ocr_recognized
    combined_rows = []
    # Build a per-image map of OCR recognized counts
    ocr_by_image = {}
    if ocr and 'details' in ocr:
        for d in ocr['details']:
            img = Path(d['image']).name
            ocr_by_image.setdefault(img, {'total':0,'recognized':0})
            ocr_by_image[img]['total'] += 1
            if d.get('ocr_text'):
                ocr_by_image[img]['recognized'] += 1

    for per in cat.get('per_image', []):
        img = Path(per['image']).name
        # aggregate tp/fp/fn for this image
        tp = fp = fn = 0
        for class_id, counts in per['counts'].items():
            tp += counts.get('tp',0)
            fp += counts.get('fp',0)
            fn += counts.get('fn',0)
        # micro precision/recall/f1 for this image
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        ocr_stats = ocr_by_image.get(img, {'total':0,'recognized':0})
        combined_rows.append({
            'image': img,
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': prec, 'recall': rec, 'f1': f1,
            'ocr_crops': ocr_stats['total'], 'ocr_recognized': ocr_stats['recognized']
        })

    save_csv(combined_rows, outdir / f'full_details_{timestamp}.csv')

    # Attempt to import PlateDetector from plate_detector.py
    try:
        from plate_detector import PlateDetector
        detector = PlateDetector(config.PLATE_MODEL_PATH, use_tesseract=False)
        print('Running OCR evaluation (using PlateDetector)...')
        ocr = run_ocr_eval(detector, 'bikehelmetnumberplate/val/images', args.limit)
        save_json(ocr, outdir / f'ocr_eval_{timestamp}.json')
        save_csv(ocr['details'], outdir / f'ocr_details_{timestamp}.csv')
    except Exception as e:
        print('PlateDetector not available or failed:', e, file=sys.stderr)
        ocr = None

    # Aggregate summary and write METRICS_LATEST
    summary = {
        'timestamp': timestamp,
        'category_micro_f1': cat['micro']['f1'],
        'category_micro_precision': cat['micro']['precision'],
        'category_micro_recall': cat['micro']['recall'],
        'ocr_summary': ocr if ocr is not None else {},
    }

    save_json(summary, outdir / 'summary_latest.json')

    # Also write a simple METRICS_LATEST.txt
    latest_txt = Path('METRICS_LATEST.txt')
    with latest_txt.open('w', encoding='utf-8') as f:
        f.write(f"micro_average_f1: {summary['category_micro_f1']*100:.1f}%\n")
        f.write(f"date: {timestamp}\n")

    print('Evaluation complete. Outputs saved to', outdir)


if __name__ == '__main__':
    main()
