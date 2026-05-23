import os
import cv2
import numpy as np
from plate_detector import PlateDetector
from esrgan import super_resolve, get_sr_runtime_info
from plate_normalizer import is_valid_indian_plate


def find_images(root_dirs, extensions=None, max_files=3):
    if extensions is None:
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    found = []
    for root in root_dirs:
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if os.path.splitext(fn)[1].lower() in extensions:
                    found.append(os.path.join(dirpath, fn))
                    if len(found) >= max_files:
                        return found
    return found


def best_from_results(results):
    if not results:
        return None, 0.0
    # results: list of (text, conf, status)
    best = max(results, key=lambda x: x[1])
    return best[0], float(best[1])


def compare_on_images(folders, limit=3):
    imgs = find_images(folders, max_files=limit)
    if not imgs:
        print('No images found')
        return

    pd = PlateDetector(os.environ.get('PLATE_MODEL_PATH', 'models/number_plate_model.pt'), use_gpu=False)

    # Warm up SR once so weights are downloaded/loaded before reporting.
    _ = super_resolve(np.zeros((20, 40, 3), dtype=np.uint8), target_height=80, use_realesrgan=True)
    info = get_sr_runtime_info()
    print("\nSR runtime info:")
    print(f"  realesrgan_ready: {info['realesrgan_ready']}")
    print(f"  device: {info['device']}")
    print(f"  weight_path: {info['weight_path']}")
    print(f"  weight_exists: {info['weight_exists']}")
    print(f"  weight_size_mb: {info['weight_size_mb']}")
    if info.get("last_error"):
        print(f"  last_error: {info['last_error']}")

    total_boxes = 0
    improved_conf = 0
    worsened_conf = 0
    same_conf = 0
    orig_valid_count = 0
    sr_valid_count = 0
    orig_conf_sum = 0.0
    sr_conf_sum = 0.0

    for img_path in imgs:
        print('\n== Image:', img_path)
        img = cv2.imread(img_path)
        if img is None:
            print('  Failed to read')
            continue

        # run YOLO detection to get plate boxes
        results = pd.model(img, conf=0.20, verbose=False)
        boxes = results[0].boxes if results[0].boxes is not None else []
        if not boxes:
            print('  No boxes detected')
            continue

        for i, box in enumerate(boxes[:3], start=1):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img[max(0, y1):min(img.shape[0], y2), max(0, x1):min(img.shape[1], x2)]
            if crop.size == 0:
                continue

            # Without SR: use preprocess_variants + OCR wrappers
            variants = pd.preprocess_variants(crop)
            orig_results = pd.easyocr_on_variants(variants)
            if not orig_results and pd.use_tess:
                orig_results = pd.tesseract_on_variants(variants)

            orig_text, orig_conf = best_from_results(orig_results)

            # With SR: upscale then OCR
            sr_crop = super_resolve(crop, target_height=150, use_realesrgan=True)
            variants_sr = pd.preprocess_variants(sr_crop)
            sr_results = pd.easyocr_on_variants(variants_sr)
            if not sr_results and pd.use_tess:
                sr_results = pd.tesseract_on_variants(variants_sr)

            sr_text, sr_conf = best_from_results(sr_results)

            valid_orig = is_valid_indian_plate(orig_text)[0] if orig_text else False
            valid_sr = is_valid_indian_plate(sr_text)[0] if sr_text else False

            total_boxes += 1
            orig_conf_sum += orig_conf
            sr_conf_sum += sr_conf
            if sr_conf > orig_conf:
                improved_conf += 1
            elif sr_conf < orig_conf:
                worsened_conf += 1
            else:
                same_conf += 1

            if valid_orig:
                orig_valid_count += 1
            if valid_sr:
                sr_valid_count += 1

            print(f'  Box {i}: size={crop.shape[:2]}')
            print(f'    Without SR -> text={orig_text} conf={orig_conf:.2f} valid={valid_orig}')
            print(f'    With SR    -> text={sr_text} conf={sr_conf:.2f} valid={valid_sr}')

    if total_boxes > 0:
        avg_orig = orig_conf_sum / total_boxes
        avg_sr = sr_conf_sum / total_boxes
        print("\n=== QUICK COMPARISON SUMMARY ===")
        print(f"boxes_compared: {total_boxes}")
        print(f"avg_conf_without_sr: {avg_orig:.3f}")
        print(f"avg_conf_with_sr:    {avg_sr:.3f}")
        print(f"avg_conf_delta:      {(avg_sr - avg_orig):+.3f}")
        print(f"conf_improved_boxes: {improved_conf}")
        print(f"conf_worsened_boxes: {worsened_conf}")
        print(f"conf_same_boxes:     {same_conf}")
        print(f"valid_without_sr:    {orig_valid_count}")
        print(f"valid_with_sr:       {sr_valid_count}")


if __name__ == '__main__':
    base = os.path.dirname(os.path.abspath(__file__))
    folders = [
        os.path.join(base, 'bikehelmetnumberplate'),
        os.path.join(base, 'numberplate')
    ]
    compare_on_images(folders, limit=3)
