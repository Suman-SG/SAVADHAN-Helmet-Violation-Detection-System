import argparse
import cv2
import os
import sys

from plate_detector import PlateDetector, validate_state_code
import config


def main():
    p = argparse.ArgumentParser(description="Extract number plate(s) from an image (no email, no fines).")
    p.add_argument("image", nargs="?", help="Path to input image")
    p.add_argument("--save", help="Save annotated image to this path", default=None)
    p.add_argument("--use-gpu", action="store_true", help="Enable GPU for OCR (if available)")
    args = p.parse_args()

    img_path = args.image
    if not img_path:
        default_path = os.path.join("images", "new100.jpg")
        prompt = f"Enter image path [{default_path}]: "
        entered = input(prompt).strip()
        img_path = entered or default_path

    if not os.path.exists(img_path):
        print(f"ERROR: image not found: {img_path}")
        sys.exit(2)

    img = cv2.imread(img_path)
    if img is None:
        print(f"ERROR: failed to read image: {img_path}")
        sys.exit(2)

    # Initialize detector using model path from config
    detector = PlateDetector(config.PLATE_MODEL_PATH, use_gpu=args.use_gpu, use_tesseract=True)

    print(f"Processing: {os.path.basename(img_path)}")
    plates = detector.detect(img)

    if not plates:
        print("No plates detected.")
        return

    for i, plate in enumerate(plates, start=1):
        text = plate.get("text") or "NOT_DETECTED"
        ocr_conf = plate.get("ocr_conf", 0.0)
        det_conf = plate.get("detection_conf", 0.0)
        status = plate.get("format_status", "INVALID")
        print(f"Plate {i}: {text} | ocr_conf={ocr_conf:.2f} | detect_conf={det_conf:.2f} | status={status}")

    def candidate_score(item):
        text = (item.get("text") or "").upper()
        ocr_conf = float(item.get("ocr_conf", 0.0))
        det_conf = float(item.get("detection_conf", 0.0))
        status = item.get("format_status", "INVALID")

        status_bonus = {
            "FULL": 0.30,
            "PARTIAL": 0.15,
            "RAW": 0.05,
            "INVALID": 0.0,
        }.get(status, 0.0)

        state_bonus = 0.08 if validate_state_code(text) else 0.0
        length_bonus = 0.05 if len(text) >= 8 else 0.0

        # Penalize candidates that were accepted only via confusion-mapping
        fixed = bool(item.get("fixed", False))
        fixed_penalty = 0.0
        if fixed and not (ocr_conf >= 0.80 or det_conf >= 0.50):
            fixed_penalty = -0.25

        return (0.62 * ocr_conf) + (0.20 * det_conf) + status_bonus + state_bonus + length_bonus + fixed_penalty

    # Require a minimum OCR or detection confidence to accept a candidate
    MIN_OCR_ACCEPT = 0.45
    MIN_DET_ACCEPT = 0.25

    candidates = [p for p in plates if p.get("text")]
    # Filter out low-confidence or weakly-fixed candidates
    candidates_filtered = [
        p for p in candidates
        if (float(p.get("ocr_conf", 0.0)) >= MIN_OCR_ACCEPT) or (float(p.get("detection_conf", 0.0)) >= MIN_DET_ACCEPT)
    ]

    # Further remove weak fixed mappings
    candidates_filtered = [
        p for p in candidates_filtered
        if not (p.get("fixed", False) and float(p.get("ocr_conf", 0.0)) < 0.80 and float(p.get("detection_conf", 0.0)) < 0.50)
    ]

    if candidates_filtered:
        best_plate = max(candidates_filtered, key=candidate_score)
    else:
        # No trustworthy candidate
        best_plate = None
    print("Best candidate:")
    if best_plate is None:
        print("  NOT_DETECTED | No high-confidence plate found")
    else:
        print(
            f"  {best_plate.get('text') or 'NOT_DETECTED'} "
            f"| ocr_conf={float(best_plate.get('ocr_conf', 0.0)):.2f} "
            f"| detect_conf={float(best_plate.get('detection_conf', 0.0)):.2f} "
            f"| status={best_plate.get('format_status', 'INVALID')}"
        )

    # Optionally save annotated image
    if args.save:
        out = detector.draw(img, plates)
        cv2.imwrite(args.save, out)
        print(f"Annotated image saved: {args.save}")


if __name__ == "__main__":
    main()
