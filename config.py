"""
CONFIGURATION FILE - EDIT PATHS HERE FIRST
"""
import os
import urllib.request
from dotenv import load_dotenv

load_dotenv(override=True)  # Load from .env file and override stale process vars

# ============================================================
# PATHS - UPDATE THESE
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ensure Ultralytics has a writable settings directory (prevents "not writable" warnings)
# You can override by setting YOLO_CONFIG_DIR in your environment or .env
YOLO_CONFIG_DIR = os.getenv("YOLO_CONFIG_DIR", os.path.join(BASE_DIR, "ultralytics_config"))
os.makedirs(YOLO_CONFIG_DIR, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = YOLO_CONFIG_DIR

# Model paths (CHANGE THESE to your actual paths)
HELMET_MODEL_PATH = r"C:\Users\shonu\Desktop\helmet_system\models\best.pt"  # Update this
PLATE_MODEL_PATH = r"C:\Users\shonu\Desktop\helmet_system\models\number_plate_model.pt"  # Update this

# Database
DATABASE_PATH = os.path.join(BASE_DIR, "database", "traffic.db")

# Output directories
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
ANNOTATED_DIR = os.path.join(OUTPUT_DIR, "annotated")
EVIDENCE_DIR = os.path.join(OUTPUT_DIR, "evidence")
INVOICES_DIR = os.path.join(BASE_DIR, "invoices")
CSV_PATH = os.path.join(OUTPUT_DIR, "violations.csv")

# ============================================================
# DETECTION THRESHOLDS
# ============================================================
HELMET_CONF = 0.20
HELMET_NMS = 0.45
HELMET_NOHELMET_MIN_CONF = 0.30
HELMET_NOHELMET_MARGIN = 0.10
PLATE_CONF = 0.10
PLATE_CONF_RETRY = 0.08  # Second pass on zoomed image (very lenient)
PLATE_RETRY = 0.10
PLATE_RETRY_MAX = 3
OCR_MIN_CONF = 0.30
TESS_FALLBACK_THRESH = 0.40  # Use Tesseract if EasyOCR confidence < this

# Plate acceptance thresholds
PLATE_OCR_ACCEPT = 0.40  # Minimum OCR confidence to accept a plate in pipeline
PLATE_DET_ACCEPT = 0.20  # Minimum detection confidence to accept a plate in pipeline
PLATE_OCR_FALLBACK_ACCEPT = 0.65  # Accept OCR-only plates when OCR >= this

# FAST / TEST MODES
FAST_MODE = os.getenv("FAST_MODE", "True").lower() == "true"
TEST_MODE = os.getenv("TEST_MODE", "False").lower() == "true"
TEST_EMAIL = os.getenv("TEST_EMAIL", "suman15sep2004@gmail.com")
# ============================================================
# HELMET-RIDER MATCHING
# ============================================================
MATCH_IOU = 0.15
HEAD_FRAC = 0.60
HEAD_MARGIN = 0.30
HORIZ_TOL = 80

# ============================================================
# PLATE SEARCH (pixels around rider)
# ============================================================
PLATE_PAD_X = 100
PLATE_PAD_Y = 130

# ============================================================
# NIGHT DETECTION
# ============================================================
NIGHT_BRIGHTNESS_THRESH = 80
NIGHT_STD_THRESH = 40
CLAHE_CLIP = 2.5
CLAHE_TILE = (8, 8)

# ============================================================
# DISPLAY
# ============================================================
DISPLAY_W = 900
DISPLAY_H = 600

# ============================================================
# FINE AMOUNT (₹)
# ============================================================
FINE_AMOUNT = 1000.0

# ============================================================
# EMAIL CONFIGURATION - USE ENVIRONMENT VARIABLES
# ============================================================
SMTP_CONFIG = {
    "server": os.getenv("SMTP_SERVER", "smtp.gmail.com"),
    "port": int(os.getenv("SMTP_PORT", "587")),
    "username": os.getenv("SMTP_USER", "your-email@gmail.com"),
    "password": os.getenv("SMTP_PASSWORD", "").replace(" ", ""),  # Gmail app passwords are often copied with spaces
    "from_email": os.getenv("FROM_EMAIL", os.getenv("SMTP_USER", "Traffic Violation System <your-email@gmail.com>"))
}

# Optional backup SMTP (used when primary fails)
BACKUP_SMTP_CONFIG = {
    "server": os.getenv("BACKUP_SMTP_SERVER", os.getenv("SMTP_SERVER", "smtp.gmail.com")),
    "port": int(os.getenv("BACKUP_SMTP_PORT", os.getenv("SMTP_PORT", "587"))),
    "username": os.getenv("BACKUP_SMTP_USER", ""),
    "password": os.getenv("BACKUP_SMTP_PASSWORD", "").replace(" ", ""),
    "from_email": os.getenv("BACKUP_FROM_EMAIL", os.getenv("FROM_EMAIL", ""))
}

# Email sending toggle
SEND_EMAIL = os.getenv("SEND_EMAIL", "False").lower() == "true"

# ============================================================
# GPU
# ============================================================
USE_GPU = True


def ensure_dirs():
    """Create all required directories"""
    for d in [OUTPUT_DIR, ANNOTATED_DIR, EVIDENCE_DIR, INVOICES_DIR, 
              os.path.join(BASE_DIR, "database")]:
        os.makedirs(d, exist_ok=True)


def _download_file(url: str, dest: str) -> bool:
    """Download a file from `url` to `dest`. Returns True on success."""
    try:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        print(f"Downloading model from {url} -> {dest}")
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        print(f"Model download failed: {e}")
        return False


def ensure_model_files():
    """Ensure model weight files exist locally; download if URLs provided in env.

    Environment variables (optional): `HELMET_MODEL_URL`, `PLATE_MODEL_URL`.
    If the local paths `HELMET_MODEL_PATH` or `PLATE_MODEL_PATH` already exist,
    this function does nothing for that file.
    """
    ensure_dirs()

    helmet_url = os.getenv("HELMET_MODEL_URL", "").strip()
    plate_url = os.getenv("PLATE_MODEL_URL", "").strip()

    if not os.path.exists(HELMET_MODEL_PATH):
        if helmet_url:
            _download_file(helmet_url, HELMET_MODEL_PATH)
        else:
            print("Warning: HELMET model missing and HELMET_MODEL_URL not set.")

    if not os.path.exists(PLATE_MODEL_PATH):
        if plate_url:
            _download_file(plate_url, PLATE_MODEL_PATH)
        else:
            print("Warning: PLATE model missing and PLATE_MODEL_URL not set.")