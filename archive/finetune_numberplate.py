#!/usr/bin/env python
"""
Fine-tune plate detector on number-plate-only dataset (300-400 images, 10 epochs).
Filters existing bikehelmetnumberplate dataset to use class 3 (number plate) only.
Expects: train/images and train/labels with YOLO format annotations.
"""

import os
import shutil
import glob
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import config

# Dataset paths
SOURCE_TRAIN_IMAGES = r"C:\Users\shonu\Desktop\helmet_system\bikehelmetnumberplate\train\images"
SOURCE_TRAIN_LABELS = r"C:\Users\shonu\Desktop\helmet_system\bikehelmetnumberplate\train\labels"
SOURCE_VAL_IMAGES = r"C:\Users\shonu\Desktop\helmet_system\bikehelmetnumberplate\val\images"
SOURCE_VAL_LABELS = r"C:\Users\shonu\Desktop\helmet_system\bikehelmetnumberplate\val\labels"

# Output paths
FINETUNE_DIR = "finetune_numberplate_dataset"
OUTPUT_TRAIN = os.path.join(FINETUNE_DIR, "images", "train")
OUTPUT_TRAIN_LABELS = os.path.join(FINETUNE_DIR, "labels", "train")
OUTPUT_VAL = os.path.join(FINETUNE_DIR, "images", "val")
OUTPUT_VAL_LABELS = os.path.join(FINETUNE_DIR, "labels", "val")

PLATE_CLASS_ID = 3  # "number plate" class
MAX_IMAGES = 400  # Target 300-400 images


def create_directories():
    """Create output directory structure."""
    for d in [OUTPUT_TRAIN, OUTPUT_TRAIN_LABELS, OUTPUT_VAL, OUTPUT_VAL_LABELS]:
        os.makedirs(d, exist_ok=True)
    print(f"✓ Created dataset directories in {FINETUNE_DIR}/")


def has_plate_annotation(label_file: str) -> bool:
    """Check if label file contains class 3 (number plate)."""
    if not os.path.exists(label_file):
        return False
    try:
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts and int(parts[0]) == PLATE_CLASS_ID:
                    return True
    except:
        pass
    return False


def filter_plate_annotations(label_file: str, output_file: str):
    """Extract only class 3 (number plate) annotations and remap to class 0."""
    try:
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        plate_lines = []
        for l in lines:
            if l.strip():
                parts = l.strip().split()
                if parts and int(parts[0]) == PLATE_CLASS_ID:
                    # Remap class 3 → 0 for single-class YOLO dataset
                    parts[0] = '0'
                    plate_lines.append(' '.join(parts) + '\n')
        
        if plate_lines:
            with open(output_file, 'w') as f:
                f.writelines(plate_lines)
            return True
    except:
        pass
    return False


def prepare_numberplate_dataset():
    """Extract 300-400 images with number plate annotations and split train/val."""
    print("\n" + "="*80)
    print("PREPARING NUMBER PLATE DATASET")
    print("="*80)
    
    create_directories()
    
    # Find all training images
    train_images = glob.glob(os.path.join(SOURCE_TRAIN_IMAGES, "*.*"))
    print(f"\n[1/3] Found {len(train_images)} total training images")
    
    # Filter images with plate annotations (and verify they exist)
    plate_images = []
    for img_path in train_images:
        base_name = os.path.basename(img_path)
        name_only = os.path.splitext(base_name)[0]
        label_path = os.path.join(SOURCE_TRAIN_LABELS, f"{name_only}.txt")
        
        # Verify source image exists and is readable
        try:
            if os.path.exists(img_path) and os.path.getsize(img_path) > 1000 and has_plate_annotation(label_path):
                plate_images.append((img_path, label_path))
        except Exception as e:
            print(f"  Skipping {base_name}: {e}")
    
    print(f"✓ Found {len(plate_images)} valid images with number plate annotations")
    
    # Limit to MAX_IMAGES
    if len(plate_images) > MAX_IMAGES:
        plate_images = plate_images[:MAX_IMAGES]
    
    print(f"✓ Using {len(plate_images)} images for fine-tuning (target: {MAX_IMAGES})")
    
    # Split into train (80%) and val (20%)
    split_idx = int(len(plate_images) * 0.8)
    train_split = plate_images[:split_idx]
    val_split = plate_images[split_idx:]
    
    print(f"✓ Split: {len(train_split)} train, {len(val_split)} val")
    
    # Copy train images and filter labels
    print(f"\n[2/3] Copying train images and filtering labels...")
    successful = 0
    failed = []
    for i, (img_path, label_path) in enumerate(train_split):
        if (i + 1) % 50 == 0 or (i + 1) == len(train_split):
            print(f"  Training: {i+1}/{len(train_split)}")
        
        try:
            base_name = os.path.basename(img_path)
            name_only = os.path.splitext(base_name)[0]
            
            # Verify source image can be read
            try:
                img = Image.open(img_path)
                img.verify()
            except Exception as e:
                failed.append((base_name, f"Corrupt image: {e}"))
                continue
            
            # Copy image
            out_img = os.path.join(OUTPUT_TRAIN, base_name)
            shutil.copy2(img_path, out_img)
            
            # Verify copied image exists
            if not os.path.exists(out_img) or os.path.getsize(out_img) < 1000:
                failed.append((base_name, "Copy failed or too small"))
                continue
            
            # Filter and copy labels (only class 0 after remapping)
            out_label = os.path.join(OUTPUT_TRAIN_LABELS, f"{name_only}.txt")
            filter_plate_annotations(label_path, out_label)
            successful += 1
        except Exception as e:
            failed.append((base_name, str(e)))
    
    # Copy val images and filter labels
    print(f"  Validation: {len(val_split)} images")
    for i, (img_path, label_path) in enumerate(val_split):
        try:
            base_name = os.path.basename(img_path)
            name_only = os.path.splitext(base_name)[0]
            
            # Verify source image can be read
            try:
                img = Image.open(img_path)
                img.verify()
            except Exception as e:
                failed.append((base_name, f"Corrupt image: {e}"))
                continue
            
            # Copy image
            out_img = os.path.join(OUTPUT_VAL, base_name)
            shutil.copy2(img_path, out_img)
            
            # Verify copied image exists
            if not os.path.exists(out_img) or os.path.getsize(out_img) < 1000:
                failed.append((base_name, "Copy failed"))
                continue
            
            # Filter and copy labels
            out_label = os.path.join(OUTPUT_VAL_LABELS, f"{name_only}.txt")
            filter_plate_annotations(label_path, out_label)
            successful += 1
        except Exception as e:
            failed.append((base_name, str(e)))
    
    print(f"✓ Successfully copied {successful}/{len(train_split)+len(val_split)} images")
    
    if failed:
        print(f"⚠ Skipped {len(failed)} images due to errors")
        for name, reason in failed[:5]:  # Show first 5
            print(f"  - {name}: {reason}")
        if len(failed) > 5:
            print(f"  ... and {len(failed)-5} more")
    
    # Create dataset YAML
    print(f"\n[3/3] Creating dataset configuration...")
    yaml_path = os.path.join(FINETUNE_DIR, "dataset.yaml")
    yaml_content = f"""path: {os.path.abspath(FINETUNE_DIR)}
train: images/train
val: images/val
nc: 1
names:
  0: plate
"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"✓ Dataset YAML created: {yaml_path}")
    print(f"\nDataset Summary:")
    print(f"  Train images: {len(train_split)}")
    print(f"  Val images: {len(val_split)}")
    print(f"  Class: number plate only")
    print(f"  Location: {FINETUNE_DIR}/")
    
    return yaml_path, len(train_split)


def fine_tune_model(yaml_path: str, epochs: int = 10):
    """Fine-tune plate detector on number plate dataset."""
    print("\n" + "=" * 80)
    print("FINE-TUNING PLATE DETECTOR")
    print("=" * 80)

    # Clear any cached data to avoid referencing stale files.
    cache_dirs = [
        os.path.join(FINETUNE_DIR, "labels", "train.cache"),
        os.path.join(FINETUNE_DIR, "labels", "val.cache"),
    ]
    for cache in cache_dirs:
        if os.path.exists(cache):
            os.remove(cache)
            print(f"✓ Cleared cache: {cache}")

    # Remove orphaned labels so YOLO does not try to load missing images.
    for labels_dir, images_dir in [
        (OUTPUT_TRAIN_LABELS, OUTPUT_TRAIN),
        (OUTPUT_VAL_LABELS, OUTPUT_VAL),
    ]:
        for label_file in glob.glob(os.path.join(labels_dir, "*.txt")):
            stem = os.path.splitext(os.path.basename(label_file))[0]
            image_exists = any(
                os.path.exists(os.path.join(images_dir, f"{stem}.{ext}"))
                for ext in ("jpg", "jpeg", "png", "bmp", "webp")
            )
            if not image_exists:
                os.remove(label_file)

    print(f"\n[1/3] Loading base model: {config.PLATE_MODEL_PATH}")
    model = YOLO(config.PLATE_MODEL_PATH)

    print(f"[2/3] Starting fine-tuning for {epochs} epochs...")
    print("      Expected time: 30-45 minutes (CPU) or 5-10 minutes (GPU)")

    import torch

    use_gpu = bool(torch.cuda.is_available())
    device = 0 if use_gpu else "cpu"
    batch_size = 4 if use_gpu else 2
    image_size = 512 if use_gpu else 640

    print(f"   Using device: {device} {'(GPU)' if use_gpu else '(CPU)'}")
    print(f"   Batch size: {batch_size}, Image size: {image_size}")

    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        patience=5,
        device=device,
        workers=0,
        single_cls=True,
        cache=False,
        verbose=True,
        save=True,
        project="runs/detect",
        name="numberplate_model_finetuned",
    )

    # Find and copy best model.
    print(f"\n[3/3] Saving fine-tuned model...")
    best_model_path = os.path.join("runs", "detect", "numberplate_model_finetuned", "weights", "best.pt")
    if os.path.exists(best_model_path):
        output_model = os.path.join("models", "improve_number_plate.pt")
        shutil.copy2(best_model_path, output_model)
        print(f"✓ Model saved: {output_model}")
        return output_model

    print("❌ Best model not found")
    return None


def test_model(model_path: str, test_images_count: int = 10):
    """Test fine-tuned model on 10 benchmark images."""
    print("\n" + "="*80)
    print("TESTING FINE-TUNED MODEL")
    print("="*80)
    
    import cv2
    from plate_detector import PlateDetector
    
    test_images = [
        "images/new100.jpg",
        "images/new139.jpg",
        "images/new136.jpg",
        "quick_train_data/new1.jpg",
        "quick_train_data/new10.jpg",
        "quick_train_data/new100.jpg",
        "quick_train_data/new101.jpg",
        "quick_train_data/new102.jpg",
        "quick_train_data/new103.jpg",
        "quick_train_data/new106.jpg",
    ]
    
    test_images = [img for img in test_images if os.path.exists(img)][:test_images_count]
    
    print(f"\nTesting on {len(test_images)} images:")
    total_detected = 0
    total_success = 0
    
    for img_path in test_images:
        print(f"\n  {img_path}:")
        img = cv2.imread(img_path)
        if img is None:
            print(f"    ❌ Could not load image")
            continue
        
        detector = PlateDetector(model_path)
        results = detector.detect(img)
        
        detected = sum(1 for r in results if r["success"])
        total_detected += detected
        
        for r in results:
            if r["success"]:
                total_success += 1
                print(f"    ✓ {r['text']} (conf: {r['ocr_conf']:.2f})")
        
        if detected == 0:
            print(f"    ✗ No plates detected")
    
    print(f"\nTest Summary:")
    print(f"  Total detections: {total_detected}")
    print(f"  Successfully read: {total_success}")
    if len(test_images) > 0:
        print(f"  Success rate: {total_success / len(test_images) * 100:.1f}%")


def main():
    print("\nFINE-TUNING NUMBER PLATE DETECTOR (10 epochs on 300-400 plate images)")
    print("="*80)
    
    # Check if source data exists
    if not os.path.exists(SOURCE_TRAIN_IMAGES):
        print(f"❌ Source images not found: {SOURCE_TRAIN_IMAGES}")
        print("   Available path: C:\\Users\\shonu\\Desktop\\helmet_system\\bikehelmetnumberplate\\train\\images")
        return
    
    if not os.path.exists(SOURCE_TRAIN_LABELS):
        print(f"❌ Source labels not found: {SOURCE_TRAIN_LABELS}")
        print("   Please check the dataset structure")
        return
    
    # Prepare dataset
    yaml_path, img_count = prepare_numberplate_dataset()
    
    if img_count < 100:
        print(f"⚠️  Warning: Only {img_count} images found. Consider using additional dataset if available.")
        response = input("Continue? (y/n): ").lower()
        if response != 'y':
            return
    
    # Fine-tune
    print("\n" + "="*80)
    print("Starting fine-tuning process...")
    print("="*80)
    
    model_path = fine_tune_model(yaml_path, epochs=5)
    
    if model_path:
        # Test
        test_model(model_path, test_images_count=10)
        
        print("\n" + "="*80)
        print("✓ FINE-TUNING COMPLETE")
        print("="*80)
        print(f"Fine-tuned model: {model_path}")
        print(f"\nTo use this model, update config.py:")
        print(f"  PLATE_MODEL_PATH = r'{os.path.abspath(model_path)}'")
    else:
        print("\n❌ Fine-tuning failed")


if __name__ == "__main__":
    main()
