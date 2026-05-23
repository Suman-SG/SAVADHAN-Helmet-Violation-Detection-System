# QUICK START - Fix Your Environment

## Problem
- Virtual environment (`venv`) was incomplete
- Model files missing (`models/best.pt` and `models/number_plate_model.pt`)

## Solution - Run These Commands

### Step 1: Recreate Virtual Environment
```powershell
cd C:\Users\shonu\Desktop\helmet_system
py -3 -m venv venv
Set-ExecutionPolicy -Scope Process RemoteSigned
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2: Setup Models
```powershell
python setup_models.py
```
This will download YOLOv8 models automatically.

### Step 3: Run Web App
```powershell
python web_app.py
```
Then open: http://localhost:5000

---

## If You Already Have Model Files
Place them here:
- `C:\Users\shonu\Desktop\helmet_system\models\best.pt`
- `C:\Users\shonu\Desktop\helmet_system\models\number_plate_model.pt`

Then run: `python web_app.py`

---

## Environment Variables (Optional)
Edit `.env` file to customize:
- `ADMIN_USERNAME` - default: admin
- `ADMIN_PASSWORD` - default: 2006
- `SMTP_*` - for email notifications
- `HELMET_MODEL_URL` - custom helmet model download URL
- `PLATE_MODEL_URL` - custom plate model download URL

---

## Troubleshooting

### "Scripts/Activate.ps1 not found"
→ Recreate venv with: `py -3 -m venv venv`

### Models still missing after setup_models.py
→ Check internet connection or manually download from:
- https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

### Port 5000 already in use
→ Use: `python web_app.py --port 8000`

### Tesseract OCR errors
→ Install from: https://github.com/UB-Mannheim/tesseract/wiki
