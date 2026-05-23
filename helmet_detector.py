"""
FILE: helmet_detector.py - IMPROVED VERSION WITH BETTER PREPROCESSING
═════════════════════════════════════════════════════════════════════════════
Enhanced helmet detection with:
- 1.5x upscaling + sharpening for small object detection
- Smart helmet_near_rider() function with IOU + spatial validation  
- Confidence-aware classification (weak no-helmet doesn't override helmet)
- Video-ready structure
═════════════════════════════════════════════════════════════════════════════
"""

import cv2
import numpy as np
from ultralytics import YOLO
import config

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


def iou(boxA, boxB):
    """Calculate Intersection over Union between two boxes."""
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    areaA = (ax2 - ax1) * (ay2 - ay1)
    areaB = (bx2 - bx1) * (by2 - by1)
    return inter / (areaA + areaB - inter + 1e-6)


def helmet_near_rider(helmet_box, rider_box, iou_thresh=0.1):
    """
    Smart helmet-to-rider matching:
    - Must overlap rider box (even slightly), OR
    - Its bottom edge in top 50% of rider box (head region)
    - Horizontally centered within rider + margin
    """
    hx1, hy1, hx2, hy2 = helmet_box
    rx1, ry1, rx2, ry2 = rider_box
    rider_height = ry2 - ry1

    # Check overlap
    if iou(helmet_box, rider_box) > iou_thresh:
        return True

    # Check if helmet center is horizontally within rider x range
    hcx = (hx1 + hx2) / 2
    if not (rx1 - 30 <= hcx <= rx2 + 30):
        return False

    # Check if helmet bottom is in top 50% of rider box (head region)
    head_zone_bottom = ry1 + rider_height * 0.5
    if hy2 <= head_zone_bottom and hy1 >= ry1 - rider_height * 0.5:
        return True

    return False


class HelmetDetector:
    def __init__(self, model_path: str):
        _allow_legacy_ultralytics_checkpoint()
        self.model = YOLO(model_path)
        self.use_gpu = bool(config.USE_GPU and TORCH_OK and torch.cuda.is_available())
        self.device = "cuda:0" if self.use_gpu else "cpu"
        self.class_names = {k: v.lower() for k, v in self.model.names.items()}
        print(f"[HelmetDetector] loaded | classes: {self.model.names}")
        print(f"[HelmetDetector] device: {'cuda:0' if self.use_gpu else 'cpu'}")

    def _sharpen(self, frame: np.ndarray) -> np.ndarray:
        """Apply sharpening filter to enhance helmet/rider edges."""
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(frame, -1, kernel)

    def _enhance_low_light(self, frame: np.ndarray) -> np.ndarray:
        """Boost contrast for dim scenes before running the detector."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
        l = clahe.apply(l)
        boosted = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        boosted = cv2.convertScaleAbs(boosted, alpha=1.08, beta=8)
        return boosted

    def _preprocess(self, frame: np.ndarray, brightness_pct: float = 100.0) -> np.ndarray:
        """Preprocess frame with adaptive upscale and contrast boost for small or dark images."""
        min_side = min(frame.shape[:2])
        upscale = 1.5

        if min_side < 260:
            upscale = 2.5
        elif min_side < 360:
            upscale = 2.0

        if brightness_pct < 15:
            upscale = max(upscale, 2.5)
        elif brightness_pct < 30:
            upscale = max(upscale, 2.0)

        if brightness_pct < 35:
            frame = self._enhance_low_light(frame)

        frame = cv2.resize(frame, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
        frame = self._sharpen(frame)
        return frame

    def _scale_box_to_original(self, box, scale_x, scale_y):
        """Scale box from resized coordinates back to original image."""
        x1, y1, x2, y2 = box
        return (
            int(x1 * scale_x),
            int(y1 * scale_y),
            int(x2 * scale_x),
            int(y2 * scale_y)
        )

    def detect(self, frame: np.ndarray) -> dict:
        """Main detection - returns boxes in ORIGINAL image coordinates."""
        original_h, original_w = frame.shape[:2]
        brightness_pct = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))) / 255.0 * 100.0
        min_side = min(original_h, original_w)
        
        # Preprocess (resize 1.5x for better detection of small elements)
        prep = self._preprocess(frame, brightness_pct=brightness_pct)
        prep_h, prep_w = prep.shape[:2]
        
        # Calculate scale factors to convert back to original coordinates
        scale_x = original_w / prep_w
        scale_y = original_h / prep_h
        
        print(f"  [Scale] Original: {original_w}x{original_h}, Prep: {prep_w}x{prep_h}, Scale: {scale_x:.3f}x{scale_y:.3f}")
        
        # Run YOLO inference with adaptive confidence for dark/small images.
        infer_conf = 0.20
        if brightness_pct < 15:
            infer_conf = 0.14
        elif brightness_pct < 30:
            infer_conf = 0.17

        results = self.model(
            prep,
            conf=infer_conf,
            iou=config.HELMET_NMS,
            imgsz=960 if min_side < 300 else 640,
            verbose=False,
            device=self.device,
            half=self.use_gpu,
        )

        def collect_detections(inference_results, prep_width, prep_height, stage_label: str, strict_rider_filter: bool = True):
            riders_raw_local = []
            helmets_local = []
            nohelmets_local = []
            plates_local = []
            confs_local = []

            for r in inference_results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = self.class_names.get(cls, "unknown")
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confs_local.append(conf)

                    print(f"   [{name}] {conf:.2f} [{x1},{y1},{x2},{y2}] (resized space, {stage_label})")

                    if name == "rider":
                        box_w = max(1, x2 - x1)
                        box_h = max(1, y2 - y1)
                        aspect_ratio = box_w / box_h
                        touches_edge = x1 <= 2 or y1 <= 2 or x2 >= prep_width - 2 or y2 >= prep_height - 2
                        if strict_rider_filter:
                            looks_like_false_positive = (
                                (aspect_ratio > 1.5 and conf < 0.7)
                                or (box_h < prep_height * 0.33 and box_w > prep_width * 0.5 and conf < 0.7)
                                or (touches_edge and aspect_ratio > 1.2 and conf < 0.65)
                            )
                            if looks_like_false_positive:
                                print(f"   [filtered rider] {conf:.2f} [{x1},{y1},{x2},{y2}] looks like a false positive")
                                continue
                        riders_raw_local.append((x1, y1, x2, y2))
                    elif name == "without helmet":
                        nohelmets_local.append(((x1, y1, x2, y2), conf))
                    elif name == "with helmet":
                        helmets_local.append(((x1, y1, x2, y2), conf))
                    elif "plate" in name or name == "number plate":
                        plates_local.append((x1, y1, x2, y2))

            return riders_raw_local, helmets_local, nohelmets_local, plates_local, confs_local

        riders_raw, helmets, nohelmets, plates, confs = collect_detections(results, prep_w, prep_h, "primary", strict_rider_filter=True)

        if not riders_raw and brightness_pct < 25 and min_side < 320:
            print("  [Fallback] No riders found on small dark frame, retrying with stronger enhancement...")
            fallback = self._enhance_low_light(frame)
            fallback = cv2.resize(fallback, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
            fallback = self._sharpen(fallback)
            fb_h, fb_w = fallback.shape[:2]
            fb_results = self.model(
                fallback,
                conf=0.08,
                iou=max(0.30, config.HELMET_NMS - 0.10),
                imgsz=1280,
                verbose=False,
                device=self.device,
                half=self.use_gpu,
            )
            fb_riders, fb_helmets, fb_nohelmets, fb_plates, fb_confs = collect_detections(
                fb_results, fb_w, fb_h, "fallback", strict_rider_filter=False
            )

            if fb_riders:
                riders_raw, helmets, nohelmets, plates, confs = fb_riders, fb_helmets, fb_nohelmets, fb_plates, fb_confs
                prep = fallback
                prep_h, prep_w = fb_h, fb_w
                scale_x = original_w / prep_w
                scale_y = original_h / prep_h
                print(f"  [Fallback] Accepted stronger pass: {len(riders_raw)} rider(s) found")

        # Create pillion rider from second no-helmet detection
        all_riders = list(riders_raw)
        
        if len(nohelmets) >= 2:
            # Second no-helmet is likely the pillion rider
            nh = nohelmets[1][0]
            nhx1, nhy1, nhx2, nhy2 = nh
            width = nhx2 - nhx1
            height = nhy2 - nhy1
            expanded = (nhx1 - 30, nhy1 - 40, nhx2 + 30, nhy2 + height * 2)
            all_riders.append(expanded)
            print(f"   [rider created] Pillion rider from no-helmet (resized space)")

        # Remove duplicate riders (same rider detected twice)
        unique_riders = []
        for rider in all_riders:
            is_dup = False
            for existing in unique_riders:
                if iou(rider, existing) > 0.5:
                    is_dup = True
                    break
            if not is_dup:
                unique_riders.append(rider)

        # CRITICAL: Scale ALL boxes BACK to original image size
        scaled_riders = [self._scale_box_to_original(b, scale_x, scale_y) for b in unique_riders]
        scaled_helmets = [self._scale_box_to_original(b, scale_x, scale_y) for b, _ in helmets]
        scaled_nohelmets = [self._scale_box_to_original(b, scale_x, scale_y) for b, _ in nohelmets]
        helmet_conf_boxes = [(self._scale_box_to_original(b, scale_x, scale_y), conf) for b, conf in helmets]
        nohelmet_conf_boxes = [(self._scale_box_to_original(b, scale_x, scale_y), conf) for b, conf in nohelmets]
        scaled_plates = [self._scale_box_to_original(b, scale_x, scale_y) for b in plates]

        print(f"  [Scaled] Plates found: {len(scaled_plates)}")

        # Classify each rider using smart helmet_near_rider function
        rider_status = []
        violators = []

        for rider_box in scaled_riders:
            # Check which helmet boxes match this rider
            has_helmet_match = any(helmet_near_rider(h_box, rider_box) for h_box, _ in helmet_conf_boxes)
            
            # Check which no-helmet boxes match this rider
            has_nohelmet_match = any(helmet_near_rider(nh_box, rider_box) for nh_box, _ in nohelmet_conf_boxes)

            # ━━━ CONFIDENCE-AWARE CLASSIFICATION ━━━━━━━━━━━━━━━━━━━━━━━━
            # Get the best confidence for each class
            best_helmet_conf = max([conf for h_box, conf in helmet_conf_boxes if helmet_near_rider(h_box, rider_box)], default=0.0)
            best_nohelmet_conf = max([conf for nh_box, conf in nohelmet_conf_boxes if helmet_near_rider(nh_box, rider_box)], default=0.0)

            # Decision logic:
            # - If helmet detected with good confidence → HELMET
            # - If no-helmet detected BUT with weak confidence → still treat as HELMET (conservative)
            # - If no-helmet has strong confidence → VIOLATION
            # - If nothing → UNKNOWN (treated conservatively per count)

            if has_helmet_match and best_helmet_conf > config.HELMET_CONF:
                status = "helmet"
            elif has_nohelmet_match:
                # Check if no-helmet confidence is strong enough to override
                if best_nohelmet_conf >= config.HELMET_NOHELMET_MIN_CONF:
                    status = "no_helmet"
                    violators.append(rider_box)
                else:
                    # Weak no-helmet detection - treat as safe
                    status = "helmet"
            else:
                # No helmet, no no-helmet - unknown state
                status = "unknown"
                # Conservatively: unknown is potentially unsafe (50% chance)
                if len(riders_raw) == 1:  # Only mark as violation if single rider
                    violators.append(rider_box)

            rider_status.append(status)

        # Count outcomes
        safe_count = rider_status.count("helmet")
        violation_count = len(violators)
        unknown_count = rider_status.count("unknown")

        print(f"   Final: {len(scaled_riders)} riders ({safe_count} safe, {violation_count} violations)")

        # Build result dict for compatibility
        result = {
            "riders": scaled_riders,
            "rider_status": rider_status,
            "helmet_boxes": scaled_helmets,
            "nohelmet_boxes": scaled_nohelmets,
            "plate_boxes": scaled_plates,
            "total_riders": len(scaled_riders),
            "safe_count": safe_count,
            "violation_count": violation_count,
            "has_violation": len(violators) > 0,
            "violator_boxes": violators,
            "avg_conf": round(np.mean(confs) * 100, 1) if confs else 0.0,
            "_native_hw": (original_h, original_w),
        }

        return result

    def draw(self, frame: np.ndarray, det: dict) -> np.ndarray:
        """Draw detection results on frame."""
        fh, fw = det["_native_hw"]
        display = cv2.resize(frame, (config.DISPLAY_W, config.DISPLAY_H))
        sx = config.DISPLAY_W / fw if fw > 0 else 1
        sy = config.DISPLAY_H / fh if fh > 0 else 1

        COLOR = {"helmet": (0, 200, 0), "no_helmet": (0, 0, 255), "unknown": (0, 140, 255)}
        LABEL = {"helmet": "✓ HELMET", "no_helmet": "✗ NO HELMET", "unknown": "? UNKNOWN"}

        # Draw riders
        for i, (rx1, ry1, rx2, ry2) in enumerate(det["riders"]):
            dx1, dy1 = int(rx1 * sx), int(ry1 * sy)
            dx2, dy2 = int(rx2 * sx), int(ry2 * sy)
            
            if i < len(det["rider_status"]):
                s = det["rider_status"][i]
                col = COLOR.get(s, (100, 100, 255))
                lbl = f"Rider {i+1} | {LABEL.get(s, s)}"
            else:
                col = (100, 100, 100)
                lbl = f"Rider {i+1} | UNKNOWN"
            
            cv2.rectangle(display, (dx1, dy1), (dx2, dy2), col, 2)
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(display, (dx1, dy1 - th - 6), (dx1 + tw + 4, dy1), col, -1)
            cv2.putText(display, lbl, (dx1 + 2, dy1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Draw plate boxes
        for (px1, py1, px2, py2) in det["plate_boxes"]:
            cv2.rectangle(display, (int(px1 * sx), int(py1 * sy)), 
                         (int(px2 * sx), int(py2 * sy)), (255, 255, 0), 2)

        # Draw status banner
        dh, dw = display.shape[:2]
        if det["has_violation"]:
            bc = (0, 0, 200)
            txt = f"VIOLATION | Riders:{det['total_riders']} ✓:{det['safe_count']} ✗:{det['violation_count']}"
        elif det["total_riders"] > 0:
            bc = (0, 170, 0)
            txt = f"SAFE | All {det['total_riders']} riders have helmets"
        else:
            bc = (80, 80, 80)
            txt = "NO RIDER DETECTED"

        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (dw, 38), bc, -1)
        display = cv2.addWeighted(overlay, 0.55, display, 0.45, 0)
        cv2.putText(display, txt, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return display
