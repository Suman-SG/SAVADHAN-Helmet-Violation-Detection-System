#!/usr/bin/env python
"""
Test fine-tuned plate detector and compare with original model.
Runs benchmark_plates.py with both models to measure improvement.
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime

def find_latest_training_output():
    """Find the most recent training output folder."""
    base_dir = r"C:\Users\shonu\runs\detect"
    if not os.path.exists(base_dir):
        return None
    
    folders = []
    for folder in os.listdir(base_dir):
        if folder.startswith("numberplate_model_finetuned"):
            path = os.path.join(base_dir, folder)
            if os.path.isdir(path):
                mtime = os.path.getmtime(path)
                folders.append((mtime, path))
    
    if folders:
        latest = max(folders, key=lambda x: x[0])
        return latest[1]
    return None


def check_model_exists():
    """Verify fine-tuned model was created."""
    training_output = find_latest_training_output()
    if not training_output:
        print(f"❌ ERROR: No training output folders found")
        return False, None
    
    model_path = os.path.join(training_output, "weights", "best.pt")
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Model not found at {model_path}")
        print("   Training may have failed. Check training logs.")
        return False, None
    
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"✓ Fine-tuned model found: {model_path} ({size_mb:.1f} MB)")
    return True, model_path


def copy_model_to_models_dir(src):
    """Copy best.pt to models/ for easy access."""
    dst = "models/improve_number_plate.pt"
    
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"✓ Copied to: {dst}")
        return True
    else:
        print(f"❌ Could not find source model at {src}")
        return False


def run_benchmark_comparison():
    """Run benchmark with original model, then fine-tuned model."""
    print("\n" + "="*80)
    print("BENCHMARK COMPARISON: Original vs Fine-tuned")
    print("="*80)
    
    # Ensure benchmark script exists
    if not os.path.exists("benchmark_plates.py"):
        print("❌ benchmark_plates.py not found")
        return False
    
    # Step 1: Run with original model
    print("\n[1/2] Benchmarking ORIGINAL model...")
    print("      (using models/number_plate_model.pt)")
    os.system("python benchmark_plates.py > benchmark_original.log 2>&1")
    
    if os.path.exists("benchmark_results.json"):
        with open("benchmark_results.json") as f:
            original = json.load(f)
        print(f"      ✓ Original model accuracy: {original.get('average_accuracy', 0):.1f}%")
        original_accuracy = original.get('average_accuracy', 0)
    else:
        print("      ❌ Could not find benchmark results")
        original_accuracy = 0
    
    # Step 2: Backup original, swap with fine-tuned
    print("\n[2/2] Benchmarking FINE-TUNED model...")
    print("      (using models/improve_number_plate.pt)")
    
    original_model = "models/number_plate_model.pt"
    backup_model = "models/number_plate_model_original.pt"
    finetuned_model = "models/improve_number_plate.pt"
    
    # Backup original
    if os.path.exists(original_model):
        shutil.copy2(original_model, backup_model)
        print(f"      ✓ Backed up original to: {backup_model}")
    
    # Swap with fine-tuned
    if os.path.exists(finetuned_model):
        shutil.copy2(finetuned_model, original_model)
        print(f"      ✓ Swapped to fine-tuned model")
        
        # Run benchmark
        os.system("python benchmark_plates.py > benchmark_finetuned.log 2>&1")
        
        if os.path.exists("benchmark_results.json"):
            with open("benchmark_results.json") as f:
                finetuned = json.load(f)
            print(f"      ✓ Fine-tuned accuracy: {finetuned.get('average_accuracy', 0):.1f}%")
            finetuned_accuracy = finetuned.get('average_accuracy', 0)
        else:
            print("      ❌ Could not find benchmark results")
            finetuned_accuracy = 0
        
        # Restore original
        shutil.copy2(backup_model, original_model)
        print(f"      ✓ Restored original model")
    else:
        print(f"      ❌ Fine-tuned model not found: {finetuned_model}")
        finetuned_accuracy = 0
    
    # Summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"Original model accuracy:   {original_accuracy:.1f}%")
    print(f"Fine-tuned model accuracy: {finetuned_accuracy:.1f}%")
    
    improvement = finetuned_accuracy - original_accuracy
    if improvement > 0:
        print(f"✓ IMPROVEMENT: +{improvement:.1f} percentage points ({100*improvement/original_accuracy:.1f}% relative)")
    elif improvement < 0:
        print(f"⚠ REGRESSION: {improvement:.1f} percentage points")
    else:
        print(f"= NO CHANGE")
    
    return True


if __name__ == "__main__":
    print("TESTING FINE-TUNED PLATE DETECTOR")
    print("="*80)
    
    # Check model exists
    success, model_path = check_model_exists()
    if not success:
        exit(1)
    
    # Copy to models/ directory
    if not copy_model_to_models_dir(model_path):
        exit(1)
    
    # Run comparison benchmark
    run_benchmark_comparison()
    
    print("\n✓ Testing complete!")
