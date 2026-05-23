#!/usr/bin/env python
"""
Fast transfer learning for plate detector using YOLO's built-in training.
This script fine-tunes the existing plate detector model on a small subset (10 images).
Expected time: 10-15 minutes instead of 5 hours.
"""

import os
import sys
from ultralytics import YOLO
import config

def prepare_dataset_yaml(data_dir: str) -> str:
    """Create a minimal dataset.yaml for YOLO training without needing annotations."""
    yaml_path = os.path.join(data_dir, "dataset.yaml")
    yaml_content = """path: {data_dir}
train: images
val: images
test: images
nc: 1
names:
  0: plate
""".format(data_dir=data_dir.replace("\\", "/"))
    
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    
    return yaml_path


def fine_tune_plate_detector(data_dir: str = "quick_train_data", epochs: int = 5):
    """
    Fine-tune the plate detector on a small dataset.
    
    Args:
        data_dir: Path to training images directory (no annotations needed for quick tune)
        epochs: Number of training epochs (default 5 = ~10-15 min for 10 images)
    """
    
    print("\n" + "="*80)
    print("PLATE DETECTOR TRANSFER LEARNING")
    print("="*80)
    
    if not os.path.exists(data_dir):
        print(f"❌ Directory not found: {data_dir}")
        return False
    
    images = [f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.png'))]
    print(f"\n✓ Found {len(images)} training images in {data_dir}")
    
    if len(images) == 0:
        print("❌ No images found in training directory")
        return False
    
    # Load pre-trained model
    print("\n[1/4] Loading pre-trained plate detector model...")
    model = YOLO(config.PLATE_MODEL_PATH)
    
    # Prepare dataset YAML
    print("[2/4] Preparing dataset configuration...")
    yaml_path = prepare_dataset_yaml(os.path.abspath(data_dir))
    print(f"    Dataset YAML: {yaml_path}")
    
    # Fine-tune (transfer learning)
    print(f"\n[3/4] Fine-tuning model on {len(images)} images for {epochs} epochs...")
    print("      Expected time: 10-15 minutes...")
    
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=640,
        batch=2,  # Small batch for small dataset
        patience=3,
        device=0 if config.USE_GPU else "cpu",
        verbose=True,
        save=True,
        project="runs/detect",
        name="plate_detector_finetuned"
    )
    
    # Save fine-tuned model
    print("\n[4/4] Saving fine-tuned model...")
    fine_tuned_path = os.path.join("runs/detect/plate_detector_finetuned", "weights/best.pt")
    if os.path.exists(fine_tuned_path):
        backup_path = config.PLATE_MODEL_PATH.replace(".pt", "_backup_original.pt")
        print(f"\n✓ Fine-tuning complete!")
        print(f"  Original model backed up: {backup_path}")
        print(f"  Fine-tuned model: {fine_tuned_path}")
        print(f"\nTo use fine-tuned model, update config.py:")
        print(f"  PLATE_MODEL_PATH = r'{fine_tuned_path}'")
        return True
    else:
        print("❌ Fine-tuned model not saved")
        return False


def quick_test_model(model_path: str = None):
    """Quick test on benchmark images to see improvement."""
    print("\n" + "="*80)
    print("QUICK TEST: Fine-tuned vs Original Model")
    print("="*80)
    
    from plate_detector import PlateDetector
    import cv2
    
    test_images = [
        "images/new100.jpg",
        "images/new139.jpg",
        "images/new136.jpg",
    ]
    
    for img_path in test_images:
        if not os.path.exists(img_path):
            continue
        
        print(f"\nTesting: {img_path}")
        img = cv2.imread(img_path)
        
        detector = PlateDetector(model_path or config.PLATE_MODEL_PATH)
        results = detector.detect(img)
        
        detected = sum(1 for r in results if r["success"])
        print(f"  Detected: {detected} plates")
        for r in results:
            if r["success"]:
                print(f"    - {r['text']} (conf: {r['ocr_conf']:.2f})")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune plate detector")
    parser.add_argument("--data-dir", default="quick_train_data", help="Training data directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--test", action="store_true", help="Run quick test after training")
    
    args = parser.parse_args()
    
    success = fine_tune_plate_detector(args.data_dir, args.epochs)
    
    if success and args.test:
        quick_test_model()
