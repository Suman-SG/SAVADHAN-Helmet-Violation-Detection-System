"""
╔══════════════════════════════════════════════════════════════════════╗
║         HELMET VIOLATION DETECTION SYSTEM  –  run_all.py            ║
║  Single file that contains ALL modules + runs the complete pipeline  ║
║  Use this if you don't want to manage separate .py files.            ║
╚══════════════════════════════════════════════════════════════════════╝

PIPELINE
  Image / Video frame
       │
       ▼
  [NightDetector]  ──  is low-light?  ──YES──►  [CLAHE]
       │                                              │
       └────────────────NO────────────────────────────┘
       │
       ▼
  [HelmetDetector] (YOLOv8)
       │
  violation?  ──NO──►  log SAFE,  save annotated image
       │
      YES
       │
       ▼
  For each violating rider:
    [PlateDetector] (YOLO + EasyOCR + Tesseract backup)
       │
       ▼
  Save evidence JPEG
  Append to violations.csv
  (Later: query DB → send email fine)
"""

# ══════════════════════════════════════════════════════════════════════
#  STANDARD IMPORTS
# ══════════════════════════════════════════════════════════════════════
"""
run_all.py - COMPLETE SINGLE FILE VERSION
Copy all the code from:
- config.py
- night_detector.py  
- helmet_detector.py
- plate_detector.py
- database.py
- fine_system.py
- main_pipeline.py

Into this single file, then run: python run_all.py
"""

# ============================================================
# COPY ALL CODE FROM ABOVE FILES HERE
# ============================================================

# [PASTE config.py code here]
# [PASTE night_detector.py code here]
# [PASTE helmet_detector.py code here]
# [PASTE plate_detector.py code here]
# [PASTE database.py code here]
# [PASTE fine_system.py code here]
# [PASTE main_pipeline.py code here]

if __name__ == "__main__":
    # Update paths in config.py section before running
    pipeline = ViolationPipeline()
    
    TEST_IMAGE = r"C:\Users\shonu\Desktop\major_project\images\thumb.jpg"
    
    if os.path.exists(TEST_IMAGE):
        pipeline.process_image(TEST_IMAGE, show=True, save=True)
    else:
        print(f"❌ Image not found: {TEST_IMAGE}")