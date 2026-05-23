"""
FILE 1: night_detector.py
═══════════════════════════════════════════════════════════════════
Detects low-light conditions and enhances using:
  1. CLAHE  (primary  – best for partial darkness)
  2. Gamma  (secondary – fast brightening for very dark images)
  3. Combined gamma + CLAHE (best general night enhancement)

Three-level detection:
  NIGHT  (brightness < 80)  → gamma lift + strong CLAHE
  DUSK   (80–120)           → mild CLAHE only
  DAY    (> 120)            → no enhancement
  LOW_CONTRAST (std < 35)   → mild CLAHE even if bright (fog/overcast)
═══════════════════════════════════════════════════════════════════
"""

import cv2
import numpy as np


class NightDetector:
    NIGHT_THRESH    = 80
    DUSK_THRESH     = 120
    STD_LOW_THRESH  = 35

    def __init__(self):
        self._clahe_strong = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self._clahe_mild   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def analyse(self, image: np.ndarray) -> dict:
        gray        = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) \
                      if len(image.shape) == 3 else image
        brightness  = float(np.mean(gray))
        std_dev     = float(np.std(gray))
        is_night    = brightness < self.NIGHT_THRESH
        is_dusk     = self.NIGHT_THRESH <= brightness < self.DUSK_THRESH
        low_con     = std_dev < self.STD_LOW_THRESH
        return {
            "is_night"         : is_night,
            "is_dusk"          : is_dusk,
            "low_contrast"     : low_con,
            "brightness"       : round(brightness, 1),
            "brightness_pct"   : round(brightness / 255 * 100, 1),
            "std_dev"          : round(std_dev, 1),
            "needs_enhancement": is_night or is_dusk or low_con,
            "mode"             : ("NIGHT" if is_night else
                                  "DUSK"  if is_dusk  else "DAY"),
        }

    def _clahe_on_lab(self, image: np.ndarray, strong: bool) -> np.ndarray:
        lab       = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b   = cv2.split(lab)
        cl        = (self._clahe_strong if strong else self._clahe_mild).apply(l)
        return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

    def _gamma(self, image: np.ndarray, gamma: float = 1.8) -> np.ndarray:
        table = np.array([(i / 255.0) ** (1.0 / gamma) * 255
                          for i in range(256)], dtype=np.uint8)
        return cv2.LUT(image, table)

    def enhance_if_needed(self, image: np.ndarray) -> tuple:
        """
        Returns (enhanced_image, was_enhanced: bool, info: dict)
        """
        info = self.analyse(image)
        if not info["needs_enhancement"]:
            info["method_used"] = "none"
            return image.copy(), False, info

        if info["is_night"]:
            # Very dark: gamma lift first, then strong CLAHE
            out = self._gamma(image, gamma=2.0)
            out = self._clahe_on_lab(out, strong=True)
            info["method_used"] = "gamma+CLAHE_strong"
        elif info["is_dusk"]:
            out = self._clahe_on_lab(image, strong=False)
            info["method_used"] = "CLAHE_mild"
        else:
            # low contrast only (fog / overcast)
            out = self._clahe_on_lab(image, strong=False)
            info["method_used"] = "CLAHE_contrast"

        # light denoise to suppress CLAHE noise
        out = cv2.bilateralFilter(out, d=5, sigmaColor=40, sigmaSpace=40)
        info["enhanced"] = True
        return out, True, info


if __name__ == "__main__":
    nd = NightDetector()
    for label, val in [("dark", 30), ("dusk", 100), ("day", 180)]:
        img = np.full((100, 100, 3), val, dtype=np.uint8)
        _, was, info = nd.enhance_if_needed(img)
        print(f"{label:5s} | {info['brightness_pct']:5.1f}% | "
              f"{info['mode']:5s} | enhanced={was} | {info.get('method_used')}")
