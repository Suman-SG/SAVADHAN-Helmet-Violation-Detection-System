import os
import json
from main_pipeline import ViolationPipeline


def run_folder(pipeline, folder):
    print(f"\n=== Running on folder: {folder} ===")
    if not os.path.isdir(folder):
        print(f"Folder not found: {folder}")
        return None
    # Walk folder recursively and process each image individually
    extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    results = []
    for root, _, files in os.walk(folder):
        for f in files:
            if not f.lower().endswith(extensions):
                continue
            path = os.path.join(root, f)
            print(f"Processing image: {path}")
            r = pipeline.process_image(path, show=False, save=False)
            if r:
                results.append(r)
    return results


def summarize(results):
    if not results:
        print("No results to summarize.")
        return
    total_images = len(results)
    total_plates = sum(len([rec for rec in r["violation_records"] if rec["plate_text"] != "NOT_DETECTED"]) for r in results)
    print(f"Processed {total_images} images, found {total_plates} plates (in violation records).")


if __name__ == '__main__':
    pipeline = ViolationPipeline()

    base = os.path.dirname(os.path.abspath(__file__))
    folders = [
        os.path.join(base, 'bikehelmetnumberplate'),
        os.path.join(base, 'numberplate')
    ]

    all_results = {}
    for f in folders:
        res = run_folder(pipeline, f)
        all_results[f] = res
        if res is not None:
            summarize(res)

    # Save a brief JSON report
    out_path = os.path.join(base, 'test_run_report.json')
    try:
        with open(out_path, 'w', encoding='utf-8') as fh:
            json.dump({k: len(v) if v else 0 for k, v in all_results.items()}, fh, indent=2)
        print(f"Report written: {out_path}")
    except Exception as e:
        print(f"Failed to write report: {e}")
