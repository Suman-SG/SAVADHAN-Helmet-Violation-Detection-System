import os
import cv2
import numpy as np

from plate_detector import PlateDetector
from esrgan import super_resolve, get_sr_runtime_info
from plate_normalizer import is_valid_indian_plate


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def find_images(root_dir, limit=10):
    found = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() in IMAGE_EXTS:
                found.append(os.path.join(dirpath, filename))
                if len(found) >= limit:
                    return found
    return found


def best_result(results):
    if not results:
        return None, 0.0, False
    text, conf, status = max(results, key=lambda item: item[1])
    valid = is_valid_indian_plate(text)[0] if text else False
    return text, float(conf), valid


def hybrid_pick(no_sr_text, no_sr_conf, no_sr_valid, sr_text, sr_conf, sr_valid):
    candidates = []
    if no_sr_text:
        candidates.append((no_sr_text, no_sr_conf, no_sr_valid, 'nosr'))
    if sr_text:
        candidates.append((sr_text, sr_conf, sr_valid, 'sr'))

    if not candidates:
        return None, 0.0, False, 'none'

    def score(item):
        text, conf, valid, origin = item
        base = float(conf)
        if valid:
            base += 1.0
        if len(text) >= 6:
            base += 0.05
        if origin == 'sr':
            base += 0.01
        return base

    text, conf, valid, origin = max(candidates, key=score)
    return text, conf, valid, origin


def run_benchmark(folder, limit=10):
    images = find_images(folder, limit=limit)
    if not images:
        print(f"No images found under: {folder}")
        return

    detector = PlateDetector(os.environ.get('PLATE_MODEL_PATH', 'models/number_plate_model.pt'), use_gpu=False)

    # warm up SR so weights are loaded before measuring
    _ = super_resolve(np.zeros((20, 40, 3), dtype=np.uint8), target_height=80, use_realesrgan=True)
    sr_info = get_sr_runtime_info()

    stats = {
        'images': 0,
        'images_with_yolo_plate': 0,
        'boxes': 0,
        'valid_without_sr': 0,
        'valid_with_sr': 0,
        'valid_hybrid': 0,
        'sr_better_conf': 0,
        'sr_worse_conf': 0,
        'sr_same_conf': 0,
        'sum_conf_without_sr': 0.0,
        'sum_conf_with_sr': 0.0,
        'sum_conf_hybrid': 0.0,
    }

    print('\n=== SR Runtime ===')
    print(f"ready: {sr_info['realesrgan_ready']}")
    print(f"device: {sr_info['device']}")
    print(f"weight_path: {sr_info['weight_path']}")
    print(f"weight_exists: {sr_info['weight_exists']}")
    print(f"weight_size_mb: {sr_info['weight_size_mb']}")
    if sr_info.get('last_error'):
        print(f"last_error: {sr_info['last_error']}")

    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            continue

        stats['images'] += 1
        yolo = detector.model(img, conf=0.20, verbose=False)
        boxes = yolo[0].boxes if yolo[0].boxes is not None else []
        if boxes:
            stats['images_with_yolo_plate'] += 1

        print(f"\nImage: {img_path}")
        print(f"  YOLO boxes: {len(boxes)}")

        for box in boxes[:3]:
            stats['boxes'] += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img[max(0, y1):min(img.shape[0], y2), max(0, x1):min(img.shape[1], x2)]
            if crop.size == 0:
                continue

            variants = detector.preprocess_variants(crop)
            no_sr_results = detector.easyocr_on_variants(variants)
            if not no_sr_results and detector.use_tess:
                no_sr_results = detector.tesseract_on_variants(variants)

            no_sr_text, no_sr_conf, no_sr_valid = best_result(no_sr_results)

            sr_crop = super_resolve(crop, target_height=150, use_realesrgan=True)
            variants_sr = detector.preprocess_variants(sr_crop)
            sr_results = detector.easyocr_on_variants(variants_sr)
            if not sr_results and detector.use_tess:
                sr_results = detector.tesseract_on_variants(variants_sr)

            sr_text, sr_conf, sr_valid = best_result(sr_results)

            hybrid_text, hybrid_conf, hybrid_valid, hybrid_origin = hybrid_pick(
                no_sr_text, no_sr_conf, no_sr_valid,
                sr_text, sr_conf, sr_valid,
            )

            stats['sum_conf_without_sr'] += no_sr_conf
            stats['sum_conf_with_sr'] += sr_conf
            stats['sum_conf_hybrid'] += hybrid_conf
            if no_sr_valid:
                stats['valid_without_sr'] += 1
            if sr_valid:
                stats['valid_with_sr'] += 1
            if hybrid_valid:
                stats['valid_hybrid'] += 1
            if sr_conf > no_sr_conf:
                stats['sr_better_conf'] += 1
            elif sr_conf < no_sr_conf:
                stats['sr_worse_conf'] += 1
            else:
                stats['sr_same_conf'] += 1

            print(f"  Box: {crop.shape[:2]} | no_sr={no_sr_text} ({no_sr_conf:.2f}, valid={no_sr_valid}) | sr={sr_text} ({sr_conf:.2f}, valid={sr_valid}) | hybrid={hybrid_text} ({hybrid_conf:.2f}, valid={hybrid_valid}, from={hybrid_origin})")

    if stats['boxes'] == 0:
        print('\nNo plate boxes detected in the sample.')
        return

    avg_no_sr = stats['sum_conf_without_sr'] / stats['boxes']
    avg_sr = stats['sum_conf_with_sr'] / stats['boxes']
    avg_hybrid = stats['sum_conf_hybrid'] / stats['boxes']

    print('\n=== SUMMARY ===')
    print(f"images_processed: {stats['images']}")
    print(f"images_with_yolo_plate: {stats['images_with_yolo_plate']}")
    print(f"plate_boxes: {stats['boxes']}")
    print(f"valid_without_sr: {stats['valid_without_sr']}")
    print(f"valid_with_sr: {stats['valid_with_sr']}")
    print(f"valid_hybrid: {stats['valid_hybrid']}")
    print(f"avg_conf_without_sr: {avg_no_sr:.3f}")
    print(f"avg_conf_with_sr: {avg_sr:.3f}")
    print(f"avg_conf_hybrid: {avg_hybrid:.3f}")
    print(f"avg_conf_delta: {(avg_sr - avg_no_sr):+.3f}")
    print(f"avg_conf_hybrid_delta: {(avg_hybrid - avg_no_sr):+.3f}")
    print(f"sr_better_conf_boxes: {stats['sr_better_conf']}")
    print(f"sr_worse_conf_boxes: {stats['sr_worse_conf']}")
    print(f"sr_same_conf_boxes: {stats['sr_same_conf']}")


if __name__ == '__main__':
    base = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(base, 'bikehelmetnumberplate', 'train', 'images')
    run_benchmark(folder, limit=10)
