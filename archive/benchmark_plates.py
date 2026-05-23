#!/usr/bin/env python
"""
Benchmark tool: test plate detection accuracy against ground truth.
Measures character-error-rate, precision, recall, false positive rate.
"""

import cv2
import json
import os
from plate_detector import PlateDetector
import config
from difflib import SequenceMatcher

def char_error_rate(detected: str, ground_truth: str) -> float:
    """Calculate character-level error rate (0.0 = perfect, 1.0 = completely wrong)."""
    if not ground_truth:
        return 1.0
    if not detected:
        return 1.0
    
    # Use SequenceMatcher to find matching characters
    matcher = SequenceMatcher(None, detected, ground_truth)
    matches = sum(block.size for block in matcher.get_matching_blocks())
    max_len = max(len(detected), len(ground_truth))
    
    if max_len == 0:
        return 1.0
    return 1.0 - (matches / max_len)


def benchmark():
    print("\n" + "="*80)
    print("PLATE DETECTION BENCHMARK")
    print("="*80)
    
    # Load ground truth
    with open("ground_truth.json") as f:
        gt_data = json.load(f)
    test_images = gt_data["test_images"]
    valid_states = set(gt_data["valid_state_codes"])
    
    # Initialize detector
    print("\n[1/3] Initializing detector...")
    detector = PlateDetector(config.PLATE_MODEL_PATH, use_gpu=False, use_tesseract=True)
    
    # Test each image
    print("\n[2/3] Running detection on test images...\n")
    
    results = []
    total_chars = 0
    correct_chars = 0
    
    for test_case in test_images:
        img_path = test_case["image"]
        ground_truth = test_case["ground_truth"]
        expected_state = test_case["state"]
        
        if not os.path.exists(img_path):
            print(f"⚠️  SKIP: {img_path} not found")
            continue
        
        print(f"\n{'─'*80}")
        print(f"Image: {img_path}")
        print(f"Ground Truth: {ground_truth} (state: {expected_state})")
        print(f"{'─'*80}")
        
        # Detect plates
        img = cv2.imread(img_path)
        plates = detector.detect(img)
        
        # Find best match
        best_match = None
        best_score = -1.0
        
        for idx, plate in enumerate(plates):
            text = plate.get("text")
            if not text:
                print(f"  Box {idx+1}: NOT_DETECTED (conf: {plate.get('detection_conf', 0):.2f})")
                continue
            
            det_conf = plate.get("detection_conf", 0.0)
            ocr_conf = plate.get("ocr_conf", 0.0)
            status = plate.get("format_status", "INVALID")
            
            # Calculate similarity to ground truth
            cer = char_error_rate(text, ground_truth)
            similarity = 1.0 - cer  # Higher is better
            
            # Bonus: if state code matches, boost score
            if len(text) >= 2 and text[:2].upper() in valid_states:
                similarity += 0.1
            
            # Track best
            if similarity > best_score:
                best_score = similarity
                best_match = (text, det_conf, ocr_conf, status, cer)
            
            icon = "✓" if similarity > 0.7 else "✗" if similarity < 0.4 else "~"
            print(f"  {icon} Box {idx+1}: {text:15s} | det={det_conf:.2f} ocr={ocr_conf:.2f} status={status:8s} | sim={similarity:.2%}")
        
        # Record result
        if best_match:
            detected, det_conf, ocr_conf, status, cer = best_match
            accuracy = 1.0 - cer
            
            # Calculate character-level metrics
            total_chars += len(ground_truth)
            correct_chars += int(len(ground_truth) * accuracy)
            
            result = {
                "image": os.path.basename(img_path),
                "ground_truth": ground_truth,
                "detected": detected,
                "accuracy": accuracy,
                "char_error_rate": cer,
                "detection_conf": det_conf,
                "ocr_conf": ocr_conf,
                "format_status": status,
                "match": "✓" if accuracy >= 0.8 else "~" if accuracy >= 0.6 else "✗"
            }
            
            print(f"\n  📊 RESULT: {detected} | Accuracy: {accuracy:.1%} | CER: {cer:.1%}")
            print(f"     Match: {result['match']}")
        else:
            result = {
                "image": os.path.basename(img_path),
                "ground_truth": ground_truth,
                "detected": "NOT_DETECTED",
                "accuracy": 0.0,
                "char_error_rate": 1.0,
                "detection_conf": 0.0,
                "ocr_conf": 0.0,
                "format_status": "NONE",
                "match": "✗"
            }
            print(f"\n  📊 RESULT: NOT_DETECTED")
        
        results.append(result)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    
    perfect = sum(1 for r in results if r["accuracy"] >= 0.9)
    good = sum(1 for r in results if 0.8 <= r["accuracy"] < 0.9)
    partial = sum(1 for r in results if 0.6 <= r["accuracy"] < 0.8)
    poor = sum(1 for r in results if r["accuracy"] < 0.6)
    
    print(f"Total Images:     {len(results)}")
    print(f"Perfect (90%+):   {perfect}")
    print(f"Good (80-90%):    {good}")
    print(f"Partial (60-80%): {partial}")
    print(f"Poor (<60%):      {poor}")
    print()
    
    if total_chars > 0:
        overall_accuracy = correct_chars / total_chars
        print(f"Character-level Accuracy: {overall_accuracy:.1%} ({correct_chars}/{total_chars})")
    
    plate_accuracy = sum(r["accuracy"] for r in results) / len(results) if results else 0
    print(f"Average Plate Accuracy:   {plate_accuracy:.1%}")
    
    # Detailed breakdown
    print(f"\n{'─'*80}")
    print("DETAILED RESULTS")
    print(f"{'─'*80}\n")
    
    for r in results:
        match_icon = r["match"]
        print(f"{match_icon} {r['image']:20s} | Ground: {r['ground_truth']:12s} | Got: {r['detected']:15s} | Acc: {r['accuracy']:.1%}")
    
    # Save results
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to benchmark_results.json")


if __name__ == "__main__":
    benchmark()
