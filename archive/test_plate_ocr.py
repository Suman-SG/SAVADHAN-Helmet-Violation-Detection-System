import cv2
import numpy as np
from plate_detector import PlateDetector

def aggressive_zoom_plate(crop, target_height=300):
    """Aggressively zoom and enhance small plates"""
    h, w = crop.shape[:2]
    
    print(f"  Original crop size: {w}x{h}")
    
    if h >= target_height:
        return crop
    
    # Scale factor
    scale = target_height / h
    new_w = int(w * scale)
    
    # Step 1: Zoom using multiple methods
    zoomed_cubic = cv2.resize(crop, (new_w, target_height), interpolation=cv2.INTER_CUBIC)
    zoomed_lanczos = cv2.resize(crop, (new_w, target_height), interpolation=cv2.INTER_LANCZOS4)
    
    # Step 2: Apply sharpening
    blur = cv2.GaussianBlur(zoomed_cubic, (0, 0), sigmaX=2.0)
    sharpened = cv2.addWeighted(zoomed_cubic, 1.8, blur, -0.8, 0)
    
    # Step 3: Contrast enhancement
    if len(sharpened.shape) == 3:
        lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        enhanced_lab = cv2.merge((l_enhanced, a, b))
        result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        result = clahe.apply(sharpened)
    
    print(f"  Zoomed crop size: {result.shape[1]}x{result.shape[0]}")
    
    return result


def invert_crop(crop):
    if crop is None or crop.size == 0:
        return crop
    return cv2.bitwise_not(crop)

# Initialize detector
detector = PlateDetector(r"C:\Users\shonu\Desktop\helmet_system\models\number_plate_model.pt", 
                         use_gpu=False, 
                         use_tesseract=True)

# Load image
image = cv2.imread(r"C:\Users\shonu\Desktop\helmet_system\images\thumb.jpg")

# Run detection
print("\nRunning plate detection...")
plates = detector.detect(image)

print(f"\nFound {len(plates)} plate detections")

for i, plate in enumerate(plates):
    print(f"\n{'='*50}")
    print(f"Plate {i+1}:")
    print(f"  BBox: {plate.get('bbox')}")
    print(f"  Detection conf: {plate.get('detection_conf')}")
    
    if plate.get('bbox'):
        x1, y1, x2, y2 = plate['bbox']
        
        # Extract crop
        crop = image[y1:y2, x1:x2]
        
        if crop.size > 0:
            # Save original crop
            cv2.imwrite(f"original_plate_{i+1}.jpg", crop)
            print(f"  Original crop saved: original_plate_{i+1}.jpg (size: {crop.shape[1]}x{crop.shape[0]})")
            
            # Aggressively zoom
            zoomed = aggressive_zoom_plate(crop, target_height=300)
            cv2.imwrite(f"zoomed_plate_{i+1}.jpg", zoomed)
            print(f"  Zoomed crop saved: zoomed_plate_{i+1}.jpg")
            
            # Try OCR on zoomed version
            print(f"\n  Trying OCR on zoomed plate...")
            result = detector._ocr_crop(zoomed)
            
            if result:
                text, conf, method, status = result
                print(f"  ✓ SUCCESS!")
                print(f"    Text: {text}")
                print(f"    Confidence: {conf:.0%}")
                print(f"    Method: {method}")
                print(f"    Status: {status}")
            else:
                print(f"  ✗ OCR still failed")

            # Try inverted version as an alternate polarity pass
            print(f"\n  Trying OCR on inverted zoomed plate...")
            inverted = invert_crop(zoomed)
            inverted_result = detector._ocr_crop(inverted)

            if inverted_result:
                text, conf, method, status = inverted_result
                print(f"  ✓ INVERTED SUCCESS!")
                print(f"    Text: {text}")
                print(f"    Confidence: {conf:.0%}")
                print(f"    Method: {method}")
                print(f"    Status: {status}")
            else:
                print(f"  ✗ Inverted OCR also failed")
                
                # Try different preprocessing
                print(f"\n  Trying alternative preprocessing...")
                
                # Convert to grayscale if needed
                if len(zoomed.shape) == 3:
                    gray = cv2.cvtColor(zoomed, cv2.COLOR_BGR2GRAY)
                else:
                    gray = zoomed.copy()
                
                # Try different thresholds
                methods = [
                    ("OTSU", cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
                    ("Adaptive", cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),
                    ("Simple (100)", cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]),
                    ("Simple (150)", cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]),
                ]
                
                for name, thresh in methods:
                    cv2.imwrite(f"plate_{name}.jpg", thresh)
                    print(f"    Saved {name} threshold: plate_{name}.jpg")