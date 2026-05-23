"""
FILE 6: main_pipeline.py - WITH AGGRESSIVE PLATE ZOOM
═══════════════════════════════════════════════════════════════════
Fixed: Aggressive zoom for small plates before OCR
═══════════════════════════════════════════════════════════════════
"""

import cv2
import csv
import os
import re
import sys
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from night_detector import NightDetector
from helmet_detector import HelmetDetector
from plate_detector import PlateDetector
from plate_normalizer import normalize_plate, validate_state_code
from database import TrafficDatabase
from fine_system import issue_fine

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False


class ViolationPipeline:
    """Complete violation detection pipeline"""
    
    CSV_HEADER = [
        "timestamp", "image_file", "total_riders", "safe_riders", 
        "violation_count", "night_image", "enhancement_applied",
        "plate_text", "plate_valid", "plate_two_row", "ocr_conf", 
        "ocr_method", "owner_name", "owner_email", "registered", 
        "violation_id", "invoice_path", "email_sent", "evidence_path"
    ]
    
    def __init__(self, suppress_emails: bool = False):
        config.ensure_dirs()
        # NOTE: Disabled automatic model downloading during initialization to
        # avoid blocking container startup on Spaces (large model downloads
        # can hang or exceed resource limits). If you need runtime download,
        # call `config.ensure_model_files()` manually or enable it behind
        # an explicit admin action.
        #
        # try:
        #     if hasattr(config, 'ensure_model_files'):
        #         config.ensure_model_files()
        # except Exception as e:
        #     print(f"Warning: ensure_model_files failed: {e}")
        self._init_csv()
        self.suppress_emails = suppress_emails
        
        print("=" * 65)
        print("  HELMET VIOLATION DETECTION SYSTEM")
        print("=" * 65)
        
        print("\n[1/4] Initializing Night Detector...")
        self.night = NightDetector()
        
        print("\n[2/4] Initializing Helmet Detector...")
        self.helmet = HelmetDetector(config.HELMET_MODEL_PATH)
        
        print("\n[3/4] Initializing Plate Detector...")
        self.plate = PlateDetector(config.PLATE_MODEL_PATH, 
                                   use_gpu=config.USE_GPU,
                                   use_tesseract=True)
        
        print("\n[4/4] Initializing Database...")
        self.db = TrafficDatabase(config.DATABASE_PATH)
        
        print("\n" + "=" * 65)
        print("  ✅ ALL MODULES READY")
        print("=" * 65 + "\n")
    
    def _init_csv(self):
        if not os.path.exists(config.CSV_PATH):
            with open(config.CSV_PATH, "w", newline="") as f:
                csv.writer(f).writerow(self.CSV_HEADER)
    
    def _write_csv(self, record):
        with open(config.CSV_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_HEADER)
            writer.writerow(record)
    
    def _aggressive_zoom_plate(self, crop, target_height=250):
        """Aggressively zoom small plates for OCR"""
        if crop is None or crop.size == 0:
            return crop
        
        h, w = crop.shape[:2]
        
        if h >= target_height:
            return crop
        
        # Calculate scale
        scale = target_height / h
        new_w = int(w * scale)
        
        print(f"        Zooming plate from {w}x{h} to {new_w}x{target_height}")
        
        # Zoom with high quality interpolation
        zoomed = cv2.resize(crop, (new_w, target_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Apply sharpening
        blur = cv2.GaussianBlur(zoomed, (0, 0), sigmaX=1.5)
        zoomed = cv2.addWeighted(zoomed, 1.6, blur, -0.6, 0)
        
        # Enhance contrast if color image
        if len(zoomed.shape) == 3:
            lab = cv2.cvtColor(zoomed, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            enhanced = cv2.merge((l_enhanced, a, b))
            zoomed = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return zoomed

    def _match_plate_to_rider(self, rider_box, plate_box):
        """Score how well a plate belongs to a rider using geometry."""
        rx1, ry1, rx2, ry2 = rider_box
        px1, py1, px2, py2 = plate_box

        rider_w = max(1, rx2 - rx1)
        rider_h = max(1, ry2 - ry1)
        plate_w = max(1, px2 - px1)
        plate_h = max(1, py2 - py1)

        cx_r = (rx1 + rx2) / 2.0
        cx_p = (px1 + px2) / 2.0
        cy_r = (ry1 + ry2) / 2.0
        cy_p = (py1 + py2) / 2.0

        dx = abs(cx_p - cx_r) / rider_w
        dy = (cy_p - cy_r) / rider_h

        horiz_ok = dx <= 0.70
        vertical_ok = -0.20 <= dy <= 0.95

        ix1 = max(rx1, px1)
        iy1 = max(ry1, py1)
        ix2 = min(rx2, px2)
        iy2 = min(ry2, py2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = rider_w * rider_h + plate_w * plate_h - inter
        iou = inter / union if union > 0 else 0.0

        score = (1.0 - min(dx, 1.0)) * 0.35 + max(0.0, 1.0 - abs(dy)) * 0.25 + min(iou * 2.0, 0.40)

        if horiz_ok and vertical_ok:
            score += 0.10

        return round(score, 3), iou, dx, dy

    def _normalize_plate_text(self, text: str) -> str:
        """Normalize plate text using `plate_normalizer.normalize_plate`.
        Returns cleaned plate or original cleaned string when normalization
        cannot produce a valid plate.
        """
        if not text:
            return text

        cleaned, status, corrected = normalize_plate(text)
        if cleaned:
            if corrected:
                print(f"    ⚙️ Normalized plate {text} -> {cleaned}")
            return cleaned

        # Fallback: return conservative cleaned uppercase string
        return re.sub(r'[^A-Za-z0-9]', '', text).upper()
    
    def _ocr_plate_with_zoom(self, crop):
        """Run OCR on plate with aggressive zoom"""
        if crop is None or crop.size == 0:
            return None
        
        try:
            attempts = [
                (crop, 0.18),
                (self._aggressive_zoom_plate(crop, target_height=250), 0.15),
            ]

            best_result = None

            for candidate_crop, min_conf in attempts:
                result = self.plate._ocr_crop(candidate_crop)
                if not result:
                    continue

                text, conf, method, status = result
                if len(text) < 4:
                    continue

                if conf >= min_conf:
                    candidate = {
                        "text": text,
                        "ocr_conf": conf,
                        "ocr_method": method,
                        "valid_format": status == "FULL"
                    }
                    if best_result is None or candidate["ocr_conf"] > best_result["ocr_conf"]:
                        best_result = candidate

            if best_result:
                return best_result
            
            return None
        except Exception as e:
            print(f"        OCR error: {e}")
            return None
    
    def _save_evidence(self, original, rider_box, plate_box, plate_text):
        """Save cropped evidence image."""
        rx1, ry1, rx2, ry2 = rider_box
        px1, py1, px2, py2 = plate_box
        
        h, w = original.shape[:2]
        
        ux1 = max(0, min(rx1, px1) - 50)
        uy1 = max(0, min(ry1, py1) - 50)
        ux2 = min(w, max(rx2, px2) + 50)
        uy2 = min(h, max(ry2, py2) + 50)
        
        crop = original[uy1:uy2, ux1:ux2].copy()
        ox, oy = ux1, uy1
        
        cv2.rectangle(crop, (rx1-ox, ry1-oy), (rx2-ox, ry2-oy), (0, 0, 255), 3)
        cv2.rectangle(crop, (px1-ox, py1-oy), (px2-ox, py2-oy), (0, 255, 0), 3)
        cv2.putText(crop, plate_text, (px1-ox, max(py1-oy-8, 12)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        safe_text = re.sub(r'[^A-Z0-9]', '', plate_text.upper())[:10]
        filename = f"evidence_{safe_text}_{timestamp}.jpg"
        filepath = os.path.join(config.EVIDENCE_DIR, filename)
        
        cv2.imwrite(filepath, crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return filepath
    
    def _save_annotated(self, annotated, source_path):
        base = os.path.splitext(os.path.basename(source_path))[0]
        timestamp = datetime.now().strftime("%H%M%S")
        filepath = os.path.join(config.ANNOTATED_DIR, f"{base}_{timestamp}.jpg")
        cv2.imwrite(filepath, annotated)
        return filepath
    
    def _display_image(self, image, title="Result"):
        if MATPLOTLIB_OK:
            plt.figure(figsize=(14, 8))
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.title(title, fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.show()
        else:
            h, w = image.shape[:2]
            if w > 1000:
                scale = 1000 / w
                display = cv2.resize(image, (int(w*scale), int(h*scale)))
            else:
                display = image
            cv2.imshow(title, display)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    def process_image(self, image_path, show=True, save=True):
        """Process a single image through the full pipeline."""
        
        print(f"\n{'─'*65}")
        print(f"  Processing: {os.path.basename(image_path)}")
        print(f"{'─'*65}")
        
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"  ❌ Cannot load: {image_path}")
            return None
        
        original = frame.copy()
        
        # STEP 1: Night enhancement
        enhanced, was_enhanced, night_info = self.night.enhance_if_needed(frame)
        print(f"  🌙 Brightness: {night_info['brightness_pct']}% | "
              f"Night: {night_info['is_night']} | Enhanced: {was_enhanced}")
        
        # STEP 2: Helmet detection
        det = self.helmet.detect(enhanced)
        print(f"  👤 Riders: {det['total_riders']} | "
              f"Safe: {det['safe_count']} | "
              f"Violations: {det['violation_count']} | "
              f"AvgConf: {det['avg_conf']}%")
        
        annotated = self.helmet.draw(enhanced, det)
        
        result = {
            "image_path": image_path,
            "night_info": night_info,
            "was_enhanced": was_enhanced,
            "detection": det,
            "violation_records": [],
            "annotated": annotated,
            "annotated_path": None
        }
        
        if not det["has_violation"]:
            print("  ✅ SAFE - No violations detected")
            if save:
                result["annotated_path"] = self._save_annotated(annotated, image_path)
            if show:
                self._display_image(annotated, "SAFE")
            return result
        
        print(f"\n  ⚠️  VIOLATION - Processing {len(det['violator_boxes'])} violator(s)")
        
        # Use dedicated plate detector in violator-focused regions for speed/precision.
        print(f"\n  📋 Running dedicated plate detector...")
        successful_plates = []
        h_img, w_img = enhanced.shape[:2]

        for idx, rider_box in enumerate(det["violator_boxes"]):
            rx1, ry1, rx2, ry2 = rider_box
            rw = max(1, rx2 - rx1)
            rh = max(1, ry2 - ry1)

            sx1 = max(0, rx1 - int(rw * 0.35))
            sx2 = min(w_img, rx2 + int(rw * 0.35))
            sy1 = max(0, ry1 + int(rh * 0.35))
            sy2 = min(h_img, ry2 + int(rh * 0.20))

            if sx2 <= sx1 or sy2 <= sy1:
                continue

            region = (sx1, sy1, sx2, sy2)
            rider_plates = self.plate.detect(enhanced, search_region=region)
            for p in rider_plates:
                if p.get("success") and p.get("text") and p.get("bbox"):
                    successful_plates.append(p)
        
        if successful_plates:
            print(f"     Found {len(successful_plates)} plate(s) from dedicated detector")
            for p in successful_plates:
                print(f"     ✓ Plate: {p['text']} (conf: {p['ocr_conf']:.0%})")
        else:
            print(f"     No plates found by dedicated detector, trying helmet model plates...")
            
            # Fallback to helmet model plates
            helmet_plates = det.get("plate_boxes", [])
            for pb in helmet_plates:
                px1, py1, px2, py2 = pb
                plate_crop = enhanced[py1:py2, px1:px2]
                if plate_crop.size > 0:
                    print(f"     Trying OCR on helmet plate (size: {plate_crop.shape[1]}x{plate_crop.shape[0]})")
                    ocr_result = self._ocr_plate_with_zoom(plate_crop)
                    if ocr_result:
                        plate_dict = {
                            "text": ocr_result["text"],
                            "bbox": (px1, py1, px2, py2),
                            "ocr_conf": ocr_result["ocr_conf"],
                            "ocr_method": ocr_result["ocr_method"],
                            "format_status": "FULL" if ocr_result["valid_format"] else "PARTIAL",
                            "success": True,
                            "detection_conf": 0.0,
                            "two_row": False
                        }
                        successful_plates.append(plate_dict)
                        print(f"     ✓ OCR successful: {ocr_result['text']}")
        
        # Process each violator
        for idx, rider_box in enumerate(det["violator_boxes"]):
            print(f"\n  ─── Violator {idx+1} ───")
            
            best_plate = None
            
            # Find plate belonging to this rider
            for p in successful_plates:
                # Validate plate structure before unpacking
                if not isinstance(p.get("bbox"), tuple) or len(p["bbox"]) != 4:
                    print(f"     ⚠️  Skipping invalid plate - missing or malformed bbox")
                    continue
                
                match_score, iou, dx, dy = self._match_plate_to_rider(rider_box, p["bbox"])
                plate_strength = p.get("ocr_conf", 0) * 0.7 + match_score * 0.3

                if match_score >= 0.32 and (best_plate is None or plate_strength > best_plate.get("match_strength", 0)):
                    best_plate = dict(p)
                    best_plate["match_strength"] = plate_strength
                    best_plate["match_score"] = match_score
                    print(f"     📋 Matched plate: {p['text']} (ocr: {p['ocr_conf']:.0%}, geom: {match_score:.2f}, iou: {iou:.2f})")
            
            # Initialize record with defaults
            record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "image_file": os.path.basename(image_path),
                "total_riders": det["total_riders"],
                "safe_riders": det["safe_count"],
                "violation_count": det["violation_count"],
                "night_image": night_info["is_night"],
                "enhancement_applied": was_enhanced,
                "plate_text": "NOT_DETECTED",
                "plate_valid": False,
                "plate_two_row": False,
                "ocr_conf": 0.0,
                "ocr_method": "none",
                "owner_name": "",
                "owner_email": "",
                "registered": False,
                "violation_id": "",
                "invoice_path": "",
                "email_sent": False,
                "evidence_path": ""
            }
            
            evidence_path = None
            normalized = "NOT_DETECTED"
            
            if best_plate:
                # Ensure required fields exist before using
                if not best_plate.get("bbox") or not best_plate.get("text"):
                    print(f"    ⚠️  Plate missing required fields, skipping")
                    best_plate = None
                else:
                    # Enforce stricter acceptance thresholds
                    ocr_conf = float(best_plate.get("ocr_conf", 0.0))
                    det_conf = float(best_plate.get("detection_conf", 0.0))

                    # Accept if both OCR and detection meet minimums
                    if ocr_conf >= config.PLATE_OCR_ACCEPT and det_conf >= config.PLATE_DET_ACCEPT:
                        pass
                    else:
                        # Allow OCR-only fallback when OCR is very confident
                        if det_conf < config.PLATE_DET_ACCEPT and ocr_conf >= config.PLATE_OCR_FALLBACK_ACCEPT:
                            print(f"    [Fallback] Accepted OCR-only plate (ocr={ocr_conf:.2f}, det={det_conf:.2f})")
                            # mark fallback acceptance (keeps best_plate)
                            best_plate["fallback_accepted"] = True
                        else:
                            print(f"    [Filter] Plate rejected due to low confidence (ocr={ocr_conf:.2f}, det={det_conf:.2f})")
                            best_plate = None

                if best_plate:
                    # Normalize OCR text conservatively (may strip trailing noise)
                    normalized = self._normalize_plate_text(best_plate["text"])

                    evidence_path = self._save_evidence(
                        original, rider_box, best_plate["bbox"], normalized
                    )

                    record["plate_text"] = normalized
                    record["plate_valid"] = (best_plate.get("format_status") == "FULL")
                    record["ocr_conf"] = best_plate.get("ocr_conf", 0.0)
                    record["ocr_method"] = best_plate.get("ocr_method", "none")
                    record["evidence_path"] = evidence_path or ""

                    print(f"    📋 Final plate: {normalized} (raw: {best_plate['text']}, conf: {best_plate['ocr_conf']:.0%})")
                    print(f"    💾 Evidence saved: {evidence_path}")
                
                # Check database
                vehicle = self.db.get_vehicle(normalized)
                if vehicle:
                    record["owner_name"] = vehicle["owner_name"]
                    record["owner_email"] = vehicle["owner_email"]
                    record["registered"] = True
                    
                    should_skip_duplicate = (not config.TEST_MODE) and self.db.has_recent_violation(normalized, hours=24)

                    if not should_skip_duplicate:
                        vid = self.db.record_violation(
                            normalized, "No Helmet", 
                            evidence_path, config.FINE_AMOUNT
                        )
                        record["violation_id"] = vid

                        recipient_email = vehicle["owner_email"]
                        if config.TEST_MODE:
                            recipient_email = config.TEST_EMAIL

                        email_enabled = config.SEND_EMAIL or config.TEST_MODE
                        if self.suppress_emails:
                            email_enabled = False

                        # Enforce per-plate cooldown: do not email same plate within configured cooldown (12 hours)
                        try:
                            last_email = self.db.get_last_email_sent_for_plate(normalized)
                            if last_email:
                                from datetime import timedelta
                                cooldown_hours = 12
                                if datetime.now() - last_email < timedelta(hours=cooldown_hours):
                                    print(f"    ⏳ Cooldown active ({cooldown_hours}h) for plate {normalized}; skipping email send")
                                    email_enabled = False
                        except Exception:
                            # If DB check fails, proceed with configured behavior
                            pass

                        fine_result = issue_fine(
                            violation_id=vid,
                            vehicle_info=vehicle,
                            violation_type="No Helmet",
                            violation_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            evidence_path=evidence_path,
                            fine_amount=config.FINE_AMOUNT,
                            send_email=email_enabled,
                            recipient_email=recipient_email,
                            test_mode=config.TEST_MODE
                        )
                        
                        record["invoice_path"] = fine_result["pdf_path"] or ""
                        record["email_sent"] = fine_result["email_sent"]
                        
                        if fine_result["email_sent"]:
                            self.db.mark_email_sent(vid)
                        
                        print(f"    ✅ Fine issued to {vehicle['owner_name']}")
                    else:
                        print(f"    ⏰ Duplicate violation - skipping fine")
                else:
                    print(f"    🧪 Demo fallback: plate not in database, sending test invoice to {config.TEST_EMAIL}")
                    demo_vehicle = {
                        "plate_number": normalized,
                        "owner_name": "Demo User",
                        "owner_email": config.TEST_EMAIL,
                    }
                    record["owner_name"] = demo_vehicle["owner_name"]
                    record["owner_email"] = demo_vehicle["owner_email"]
                    record["registered"] = False

                    vid = self.db.record_violation(
                        normalized, "No Helmet",
                        evidence_path, config.FINE_AMOUNT
                    )
                    record["violation_id"] = vid

                    fine_result = issue_fine(
                        violation_id=vid,
                        vehicle_info=demo_vehicle,
                        violation_type="No Helmet",
                        violation_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        evidence_path=evidence_path,
                        fine_amount=config.FINE_AMOUNT,
                        send_email=not self.suppress_emails,
                        recipient_email=config.TEST_EMAIL,
                        original_image_path=image_path,
                        test_mode=True
                    )

                    record["invoice_path"] = fine_result["pdf_path"] or ""
                    record["email_sent"] = fine_result["email_sent"]

                    if fine_result["email_sent"]:
                        self.db.mark_email_sent(vid)
                        print(f"    ✅ Demo email sent to {config.TEST_EMAIL}")
                    else:
                        print(f"    ⚠️  Demo email could not be sent")
            else:
                print(f"    ⚠️  No plate found for violator {idx+1}")
            
            result["violation_records"].append(record)
            self._write_csv(record)
        
        if save:
            result["annotated_path"] = self._save_annotated(annotated, image_path)
        if show:
            self._display_image(annotated, "VIOLATION DETECTED")
        
        print(f"\n  ─── SUMMARY ─────────────────────────────")
        print(f"     Riders      : {det['total_riders']}")
        print(f"     Safe        : {det['safe_count']}")
        print(f"     Violations  : {det['violation_count']}")
        for rec in result["violation_records"]:
            status = "✓" if rec['plate_text'] != "NOT_DETECTED" else "✗"
            capped_conf = min(rec['ocr_conf'], 1.0)  # Cap at 100%
            print(f"     Plate {status}: {rec['plate_text']} (conf: {capped_conf:.0%})")
        print(f"  ─────────────────────────────────────────")
        
        return result
    
    def process_batch(self, folder_path, show=False):
        extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        images = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                 if f.lower().endswith(extensions)]
        
        print(f"\n{'='*65}")
        print(f"  BATCH PROCESSING - {len(images)} images")
        print(f"{'='*65}")
        
        results = []
        for img in images:
            r = self.process_image(img, show=show, save=True)
            if r:
                results.append(r)
        
        total_violations = sum(1 for r in results if r["detection"]["has_violation"])
        total_plates = sum(len([rec for rec in r["violation_records"] if rec["plate_text"] != "NOT_DETECTED"]) for r in results)
        
        print(f"\n{'='*65}")
        print(f"  BATCH COMPLETE")
        print(f"  Images        : {len(images)}")
        print(f"  Violations    : {total_violations}")
        print(f"  Plates found  : {total_plates}")
        print(f"  CSV Report    : {config.CSV_PATH}")
        print(f"{'='*65}")
        
        return results


if __name__ == "__main__":
    pipeline = ViolationPipeline()
    
    TEST_IMAGE = r"C:\Users\shonu\Desktop\helmet_system\images\thumb.jpg"
    TEST_FOLDER = None
    
    if TEST_FOLDER and os.path.isdir(TEST_FOLDER):
        pipeline.process_batch(TEST_FOLDER)
    elif os.path.exists(TEST_IMAGE):
        pipeline.process_image(TEST_IMAGE, show=False, save=True)
    else:
        print(f"❌ Image not found: {TEST_IMAGE}")