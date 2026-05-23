"""
Simple image test runner.
Paste an image path and get the pipeline result quickly.

Usage:
  python test_image_runner.py "C:\\path\\to\\image.jpg"
  python test_image_runner.py
"""

import argparse
import os
import sys


def _apply_mode_flags(test_mode: bool):
    if test_mode:
        os.environ["TEST_MODE"] = "True"
        os.environ["SEND_EMAIL"] = "True"
        os.environ.setdefault("TEST_EMAIL", "suman15sep2004@gmail.com")


parser = argparse.ArgumentParser(description="Run the violation pipeline on one image.")
parser.add_argument("image", nargs="?", help="Path to the image to process")
parser.add_argument("--test-mode", action="store_true", help="Send demo email to TEST_EMAIL and skip 24h duplicate suppression")
args = parser.parse_args()

_apply_mode_flags(args.test_mode)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main_pipeline import ViolationPipeline


def main():
    image_path = args.image if args.image else input("Enter image path: ").strip().strip('"')
    if not image_path:
        print("No image path provided.")
        return 1

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return 1

    pipeline = ViolationPipeline()
    result = pipeline.process_image(image_path, show=False, save=True)

    if not result:
        print("No result returned.")
        return 1

    print("\nRESULT")
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Violations: {result['detection']['violation_count']}")
    if args.test_mode:
        print(f"Test email target: {os.getenv('TEST_EMAIL')}")
        print("Test mode: duplicate suppression disabled")

    for idx, record in enumerate(result.get("violation_records", []), start=1):
        print(f"Violator {idx}: {record['plate_text']} | conf={record['ocr_conf']:.0%} | email_sent={record['email_sent']}")
        if record.get("invoice_path"):
            print(f"  Invoice: {record['invoice_path']}")
        if record.get("evidence_path"):
            print(f"  Evidence: {record['evidence_path']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
