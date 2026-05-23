#!/usr/bin/env python3
"""
Setup models for helmet system.
Downloads or initializes required YOLO models.
"""
import os
import sys
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
WEIGHTS_DIR = BASE_DIR / "weights"

def ensure_dirs():
    """Create required directories."""
    MODELS_DIR.mkdir(exist_ok=True)
    WEIGHTS_DIR.mkdir(exist_ok=True)
    print(f"✓ Created directories: {MODELS_DIR}, {WEIGHTS_DIR}")

def download_file(url: str, dest: Path) -> bool:
    """Download file from URL to destination."""
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Downloading: {url}")
        print(f"  → {dest}")
        urllib.request.urlretrieve(url, dest)
        print(f"  ✓ Done ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False

def check_models():
    """Check which models are missing."""
    helmet_path = MODELS_DIR / "best.pt"
    plate_path = MODELS_DIR / "number_plate_model.pt"
    
    status = {
        "helmet": helmet_path.exists(),
        "plate": plate_path.exists()
    }
    
    print("\n📊 Model Status:")
    print(f"  Helmet Model: {'✓ EXISTS' if status['helmet'] else '✗ MISSING'} ({helmet_path})")
    print(f"  Plate Model:  {'✓ EXISTS' if status['plate'] else '✗ MISSING'} ({plate_path})")
    
    return status

def init_from_ultralytics():
    """Initialize with default YOLOv8 models from Ultralytics."""
    print("\n📥 Initializing with default YOLOv8 models...")
    print("   This will download ~200MB of pre-trained weights.\n")
    
    try:
        from ultralytics import YOLO
        
        print("1️⃣  Loading YOLOv8n (nano - for helmet detection)...")
        helmet_model = YOLO("yolov8n.pt")
        helmet_model.save(str(MODELS_DIR / "best.pt"))
        print("   ✓ Helmet model saved to models/best.pt")
        
        print("\n2️⃣  Loading YOLOv8n (nano - for plate detection)...")
        plate_model = YOLO("yolov8n.pt")
        plate_model.save(str(MODELS_DIR / "number_plate_model.pt"))
        print("   ✓ Plate model saved to models/number_plate_model.pt")
        
        print("\n✓ Models initialized successfully!")
        return True
    except ImportError:
        print("✗ Ultralytics not installed. Install with: pip install ultralytics")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    """Main setup flow."""
    print("=" * 60)
    print("  HELMET VIOLATION SYSTEM - Model Setup")
    print("=" * 60)
    
    ensure_dirs()
    status = check_models()
    
    all_exist = all(status.values())
    if all_exist:
        print("\n✓ All models present! You can run: python web_app.py")
        return 0
    
    print("\n🔧 Setup Options:")
    print("  1. Download from URLs (if configured in .env)")
    print("  2. Initialize with default YOLOv8 models (recommended)")
    print("  3. Skip (manually place .pt files in models/)")
    print("\n  Default: Option 2 (auto-download YOLOv8)")
    
    # Try Ultralytics default models
    success = init_from_ultralytics()
    
    if success:
        print("\n" + "=" * 60)
        print("  Setup Complete! Run: python web_app.py")
        print("=" * 60)
        return 0
    else:
        print("\n" + "=" * 60)
        print("  ⚠️  Setup Incomplete")
        print("=" * 60)
        print("\n📝 Manual Fix:")
        print("  1. Download YOLO models:")
        print("     - Helmet: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt")
        print("     - Plate:  (same file can be used or a custom model)")
        print("\n  2. Place in models/ folder:")
        print("     C:\\Users\\shonu\\Desktop\\helmet_system\\models\\best.pt")
        print("     C:\\Users\\shonu\\Desktop\\helmet_system\\models\\number_plate_model.pt")
        print("\n  3. Run: python web_app.py")
        return 1

if __name__ == "__main__":
    sys.exit(main())
