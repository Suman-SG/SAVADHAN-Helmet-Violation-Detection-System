import os
from ultralytics import YOLO

try:
    import torch
except ImportError:
    torch = None

model_path = r"C:\Users\shonu\Desktop\helmet_system\models\number_plate_model.pt"

# Check if file exists
if os.path.exists(model_path):
    print(f"✓ Model file exists: {model_path}")
    print(f"  File size: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")

    if torch is not None:
        cuda_ok = torch.cuda.is_available()
        print(f"  CUDA available: {cuda_ok}")
        if cuda_ok:
            print(f"  GPU count: {torch.cuda.device_count()}")
            print(f"  GPU name: {torch.cuda.get_device_name(0)}")
    
    # Try loading the model
    try:
        model = YOLO(model_path)
        print("✓ Model loaded successfully")
        print(f"  Classes: {model.names}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
else:
    print(f"✗ Model file NOT found at: {model_path}")