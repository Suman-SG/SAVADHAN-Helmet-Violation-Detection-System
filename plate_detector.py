"""
FILE: plate_detector_clean.py - SIMPLIFIED & FOCUSED
═════════════════════════════════════════════════════════════════════════════
Indian License Plate Detection + OCR with Strong Validation

Key Features:
1. Single super-resolution path (no duplicates)
2. Unified OCR voting (EasyOCR + Tesseract)
3. Indian state code validation
4. Two-row plate support
5. All thresholds from config.py

Cleaned from previous version - removed duplicate logic.
═════════════════════════════════════════════════════════════════════════════
"""

import cv2
import os
import numpy as np
import re
import shutil
from ultralytics import YOLO
import config
from plate_normalizer import normalize_plate
import math

try:
    import torch
    TORCH_OK = True
except ImportError:
    TORCH_OK = False


def _allow_legacy_ultralytics_checkpoint() -> None:
    """Allow trusted Ultralytics detection checkpoints under PyTorch 2.6+."""
    if not TORCH_OK:
        return

    try:
        from ultralytics.nn.tasks import DetectionModel

        add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
        if callable(add_safe_globals):
            add_safe_globals([DetectionModel])
    except Exception:
        pass

try:
    import easyocr
    EASYOCR_OK = True
except ImportError:
    EASYOCR_OK = False

try:
    import pytesseract
    TESSERACT_CMD = shutil.which("tesseract")
    if not TESSERACT_CMD:
        for candidate in (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ):
            if os.path.exists(candidate):
                TESSERACT_CMD = candidate
                break
    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        TESSERACT_OK = True
    else:
        TESSERACT_OK = False
except ImportError:
    TESSERACT_OK = False

try:
    from esrgan import super_resolve as esrgan_super_resolve
    ESRGAN_OK = True
except ImportError:
    ESRGAN_OK = False


class PlateDetector:
    def __init__(self, model_path: str, use_gpu: bool = False, use_tesseract: bool = True):
        _allow_legacy_ultralytics_checkpoint()
        self.model = YOLO(model_path)
        auto_gpu = bool(TORCH_OK and torch.cuda.is_available())
        self.use_gpu = bool((use_gpu or auto_gpu) and auto_gpu)
        self.device = "cuda:0" if self.use_gpu else "cpu"
        self.use_tesseract = bool(use_tesseract and TESSERACT_OK)
        
        if EASYOCR_OK:
            self.reader_easy = easyocr.Reader(['en'], gpu=self.use_gpu)
        else:
            self.reader_easy = None
            
        print("[PlateDetector] Loading YOLO model...")
        print(f"[PlateDetector] Loading EasyOCR...")
        if self.reader_easy:
            print("[PlateDetector] EasyOCR ready ✓")
        else:
            print("[PlateDetector] EasyOCR NOT available")
            
        if self.use_tesseract:
            print("[PlateDetector] Tesseract available (backup) ✓")
        else:
            print("[PlateDetector] Tesseract NOT available")
        if use_gpu and not self.use_gpu:
            print("[PlateDetector] CUDA requested but unavailable; falling back to CPU")
        print(f"[PlateDetector] device: {'cuda:0' if self.use_gpu else 'cpu'}")
            
        print("[PlateDetector] Ready.")

    def _super_resolve(self, crop: np.ndarray, target_height: int = 150) -> np.ndarray:
        """Super-resolve small plates using Real-ESRGAN or fallback"""
        h, w = crop.shape[:2]
        
        # Adaptive target: 2x scale or min 300px for tiny plates
        adaptive_target = min(300, max(150, h * 2))
        if h >= adaptive_target:
            return crop
        
        # Try Real-ESRGAN first
        if ESRGAN_OK:
            try:
                sr_result = esrgan_super_resolve(crop)
                # Cap output height to avoid memory bloat
                if sr_result.shape[0] > 400:
                    scale = 400 / sr_result.shape[0]
                    sr_result = cv2.resize(sr_result, (int(sr_result.shape[1] * scale), 400), interpolation=cv2.INTER_CUBIC)
                return sr_result
            except Exception as e:
                print(f"     [SR] ESRGAN failed: {e}, using fallback")
        
        # Fallback: bicubic interpolation with sharpening
        scale = adaptive_target / h
        new_w = int(w * scale)
        upscaled = cv2.resize(crop, (new_w, adaptive_target), interpolation=cv2.INTER_CUBIC)
        
        # Sharpen
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(upscaled, -1, kernel)
        
        return sharpened

    def _preprocess_for_ocr(self, crop: np.ndarray) -> list:
        """Generate multiple preprocessed variants for OCR voting"""
        if crop is None or crop.size == 0:
            return []
        
        variants = []
        
        # Convert to grayscale
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop.copy()
        
        # Variant 1: Denoised
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        variants.append(denoised)
        
        # Variant 2: OTSU threshold
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(otsu)
        
        # Variant 3: Adaptive threshold (best for uneven lighting)
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
        variants.append(adaptive)
        
        # Variant 4: Inverted (for white-on-dark plates)
        inverted = cv2.bitwise_not(gray)
        variants.append(inverted)

        # Variant 5: Inverted OTSU
        inverted_otsu = cv2.bitwise_not(otsu)
        variants.append(inverted_otsu)

        # Variant 6: Inverted adaptive threshold
        inverted_adaptive = cv2.bitwise_not(adaptive)
        variants.append(inverted_adaptive)
        
        return variants

    def _ocr_with_easyocr(self, image: np.ndarray) -> tuple:
        """Run EasyOCR and return (text, confidence)"""
        if not self.reader_easy:
            return None, 0.0
        
        try:
            results = self.reader_easy.readtext(image, detail=1)
            if not results:
                return None, 0.0
            
            # Combine all text
            text = ''.join(res[1] for res in results)
            conf = np.mean([res[2] for res in results]) if results else 0.0
            
            return text, conf
        except Exception as e:
            print(f"     [OCR] EasyOCR error: {e}")
            return None, 0.0

    def _prepare_tesseract_input(self, image: np.ndarray) -> np.ndarray:
        """Normalize a plate crop for single-line OCR."""
        if image is None or image.size == 0:
            return image

        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        if h < 80:
            scale = 80 / h
            gray = cv2.resize(gray, (int(w * scale), 80), interpolation=cv2.INTER_CUBIC)

        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Keep the most text-friendly polarity for Tesseract.
        if np.mean(thresh) < 127:
            thresh = cv2.bitwise_not(thresh)

        return thresh

    def _ocr_with_tesseract(self, image: np.ndarray) -> tuple:
        """Run Tesseract and return (text, confidence)"""
        if not self.use_tesseract:
            return None, 0.0
        
        try:
            prepared = self._prepare_tesseract_input(image)
            config_str = "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            result = pytesseract.image_to_data(prepared, config=config_str, output_type=pytesseract.Output.DICT)
            words = [text.strip() for text in result.get('text', []) if text and text.strip()]
            text = ''.join(words).strip()
            if not text:
                text = pytesseract.image_to_string(prepared, config=config_str).strip()
            confidences = []
            for confidence in result.get('conf', []):
                try:
                    value = float(confidence)
                except (TypeError, ValueError):
                    continue
                if value > 0:
                    confidences.append(value)

            conf = (np.mean(confidences) / 100.0) if confidences else 0.0
            text = re.sub(r'[^A-Z0-9]', '', text.upper())
            
            return text, conf
        except Exception as e:
            print(f"     [OCR] Tesseract error: {e}")
            return None, 0.0

    def _plate_quality_bonus(self, text: str) -> float:
        """Score how plate-like an OCR string looks."""
        if not text:
            return 0.0

        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
        if not cleaned:
            return 0.0

        bonus = 0.0

        # Prefer Indian plate-like prefixes.
        if re.match(r'^(BH|[A-Z]{2}[0-9]{2})', cleaned):
            bonus += 0.20

        # Strong signal for the last 4 digits that many plates end with.
        if re.search(r'[0-9]{4}$', cleaned):
            bonus += 0.18

        # Slight preference for the common letter-digit layout.
        if re.search(r'[A-Z]{1,3}[0-9]{4}$', cleaned):
            bonus += 0.10

        # Penalize very short or repetitive fragments.
        if len(cleaned) < 6:
            bonus -= 0.12

        if len(cleaned) >= 3 and cleaned[0] == cleaned[1]:
            bonus -= 0.08

        return bonus

    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """Estimate skew angle and rotate to deskew the image."""
        try:
            gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            coords = np.column_stack(np.where(th > 0))
            if coords.shape[0] < 10:
                return image
            rect = cv2.minAreaRect(coords)
            angle = rect[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle

            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return rotated
        except Exception:
            return image

    def _rotate(self, image: np.ndarray, angle: float) -> np.ndarray:
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    def _ocr_crop(self, crop: np.ndarray) -> tuple[str, float, str, str] | None:
        """OCR a single plate crop with voting"""
        if crop is None or crop.size == 0:
            return None
        
        # Generate variants
        variants = self._preprocess_for_ocr(crop)
        if not variants:
            return None

        # Deskew once and try small rotations per variant
        ocr_results = []
        rotations = [-10, -5, 0, 5, 10]

        for variant in variants:
            # Deskew variant first
            deskewed = self._deskew(variant)
            for angle in rotations:
                trial = self._rotate(deskewed, angle) if angle != 0 else deskewed

                # Try EasyOCR first
                easy_text, easy_conf = self._ocr_with_easyocr(trial)
                if easy_text and easy_conf >= config.PLATE_OCR_ACCEPT:
                    ocr_results.append(("EasyOCR", easy_text, easy_conf))

                # If EasyOCR weak, try Tesseract
                if easy_conf < config.TESS_FALLBACK_THRESH:
                    tess_text, tess_conf = self._ocr_with_tesseract(trial)
                    if tess_text and tess_conf >= config.PLATE_OCR_ACCEPT:
                        ocr_results.append(("Tesseract", tess_text, tess_conf))
        
        if not ocr_results:
            return None
        
        # Vote on best result (highest confidence + best validation)
        best_result = None
        best_score = -1
        
        for engine, text, conf in ocr_results:
            # Clean text
            cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
            
            # Validate with Indian plate rules
            normalized, status, _ = normalize_plate(cleaned)
            
            # Scoring: confidence + validation bonus + plate-shape bonus
            score = conf
            if status == "FULL":
                score += 0.5
            elif status == "PARTIAL":
                score += 0.2

            score += self._plate_quality_bonus(cleaned)
            
            if score > best_score and normalized:
                best_score = score
                best_result = (normalized, conf, engine, status)
        
        return best_result

    def _ocr_two_row(self, crop: np.ndarray) -> tuple[str, float, str, str] | None:
        """Handle two-row plates (common on bikes)"""
        h, w = crop.shape[:2]
        aspect = w / h if h > 0 else 0
        
        # Only split if roughly square-ish
        if aspect < 0.8 or aspect > 1.2:
            return None
        
        # Split horizontally
        mid_y = h // 2
        top = crop[0:mid_y, :]
        bottom = crop[mid_y:, :]
        
        # Super-resolve if small
        if h < 150:
            top = self._super_resolve(top, target_height=75)
            bottom = self._super_resolve(bottom, target_height=75)
        
        # OCR each half
        top_result = self._ocr_crop(top)
        bottom_result = self._ocr_crop(bottom)
        
        if top_result and bottom_result:
            combined = top_result[0] + bottom_result[0]
            avg_conf = (top_result[1] + bottom_result[1]) / 2
            
            # Validate combined
            normalized, status, _ = normalize_plate(combined)
            
            if normalized:
                return normalized, avg_conf, "Two-Row", status
        
        return None

    def detect(self, frame: np.ndarray, search_region: tuple = None) -> list:
        """Detect and OCR plates in frame"""
        results = []
        h, w = frame.shape[:2]
        
        if search_region:
            x1, y1, x2, y2 = search_region
            roi = frame[y1:y2, x1:x2]
        else:
            x1, y1, x2, y2 = 0, 0, w, h
            roi = frame
        
        # YOLO detection (first pass)
        detections = self.model(
            roi,
            conf=config.PLATE_CONF,
            imgsz=640,
            verbose=False,
            device=self.device,
            half=self.use_gpu,
        )

        boxes = []
        for detection in detections:
            if detection.boxes is None:
                continue
            for box in detection.boxes:
                conf = float(box.conf[0])
                x_box1, y_box1, x_box2, y_box2 = map(int, box.xyxy[0])

                # Convert to original frame coordinates
                if search_region:
                    x_box1 += x1
                    y_box1 += y1
                    x_box2 += x1
                    y_box2 += y1

                boxes.append((x_box1, y_box1, x_box2, y_box2, conf))

        # If detections are few, run multi-scale zoomed passes (lenient) to improve recall
        if len(boxes) < 2 and config.PLATE_RETRY_MAX > 0:
            scales = [1.2, 1.5]
            tries = 0
            for zoom in scales:
                if tries >= config.PLATE_RETRY_MAX:
                    break
                try:
                    zh = int(min(h, int(h * zoom)))
                    zw = int(min(w, int(w * zoom)))
                    zoomed = cv2.resize(roi, (zw, zh), interpolation=cv2.INTER_CUBIC)
                    detections2 = self.model(
                        zoomed,
                        conf=config.PLATE_CONF_RETRY,
                        imgsz=640,
                        verbose=False,
                        device=self.device,
                        half=self.use_gpu,
                    )
                    for detection in detections2:
                        if detection.boxes is None:
                            continue
                        for box in detection.boxes:
                            conf = float(box.conf[0])
                            xb1, yb1, xb2, yb2 = map(int, box.xyxy[0])

                            # Map zoomed coords back to original ROI coords
                            scale_x = w / float(zw)
                            scale_y = h / float(zh)
                            x_box1 = int(xb1 * scale_x)
                            y_box1 = int(yb1 * scale_y)
                            x_box2 = int(xb2 * scale_x)
                            y_box2 = int(yb2 * scale_y)

                            # Convert to original frame coords
                            if search_region:
                                x_box1 += x1
                                y_box1 += y1
                                x_box2 += x1
                                y_box2 += y1

                            boxes.append((x_box1, y_box1, x_box2, y_box2, conf))
                    tries += 1
                except Exception as e:
                    print(f"     [PlateDetector] Retry pass (scale={zoom}) failed: {e}")

        # Deduplicate overlapping boxes (simple IoU-based merge)
        final_boxes = []
        for bx in boxes:
            x1b, y1b, x2b, y2b, confb = bx
            keep = True
            for exist in final_boxes:
                ex1, ey1, ex2, ey2, exc = exist
                # compute IoU
                xi1 = max(x1b, ex1)
                yi1 = max(y1b, ey1)
                xi2 = min(x2b, ex2)
                yi2 = min(y2b, ey2)
                inter_w = max(0, xi2 - xi1)
                inter_h = max(0, yi2 - yi1)
                inter = inter_w * inter_h
                area1 = max(1, (x2b - x1b) * (y2b - y1b))
                area2 = max(1, (ex2 - ex1) * (ey2 - ey1))
                union = area1 + area2 - inter
                iou = inter / union if union > 0 else 0
                if iou > 0.5:
                    # keep the higher confidence box
                    if confb > exc:
                        exist[0], exist[1], exist[2], exist[3], exist[4] = x1b, y1b, x2b, y2b, confb
                    keep = False
                    break
            if keep:
                final_boxes.append([x1b, y1b, x2b, y2b, confb])

        for fb in final_boxes:
            x_box1, y_box1, x_box2, y_box2, conf = fb

            # Extract crop
            crop = frame[max(0, y_box1):min(h, y_box2), max(0, x_box1):min(w, x_box2)]

            if crop.size == 0:
                continue

            # Super-resolve if small
            if crop.shape[0] < 150:
                crop = self._super_resolve(crop, target_height=150)

            # Try OCR
            ocr_result = self._ocr_crop(crop)

            # Try two-row if single-row failed
            if not ocr_result:
                ocr_result = self._ocr_two_row(crop)

            # Build result
            if ocr_result:
                text, ocr_conf, ocr_method, status = ocr_result
                results.append({
                    "bbox": (x_box1, y_box1, x_box2, y_box2),
                    "text": text,
                    "raw": text,
                    "ocr_conf": min(ocr_conf, 1.0),  # Cap at 100%
                    "ocr_method": ocr_method,
                    "format_status": status,
                    "detection_conf": conf,
                    "success": True,
                    "two_row": ocr_method == "Two-Row"
                })
            else:
                results.append({
                    "bbox": (x_box1, y_box1, x_box2, y_box2),
                    "text": "NOT_DETECTED",
                    "raw": "",
                    "ocr_conf": 0.0,
                    "ocr_method": "none",
                    "format_status": "NONE",
                    "detection_conf": conf,
                    "success": False,
                    "two_row": False
                })

        return results
