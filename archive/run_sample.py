import os
import itertools
from main_pipeline import ViolationPipeline


def find_images(root_dirs, extensions=None, max_files=6):
    if extensions is None:
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    found = []
    for root in root_dirs:
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if os.path.splitext(fn)[1].lower() in extensions:
                    found.append(os.path.join(dirpath, fn))
                    if len(found) >= max_files:
                        return found
    return found


def run_sample(folders, limit=3):
    imgs = find_images(folders, max_files=limit)
    if not imgs:
        print("No images found in provided folders.")
        return

    pipeline = ViolationPipeline(suppress_emails=True)
    results = []
    for img in imgs:
        print(f"\n--- Processing sample image: {img}")
        res = pipeline.process_image(img, show=False, save=False)
        results.append((img, res))
    return results


if __name__ == '__main__':
    base = os.path.dirname(os.path.abspath(__file__))
    folders = [
        os.path.join(base, 'bikehelmetnumberplate'),
        os.path.join(base, 'numberplate')
    ]

    sample_results = run_sample(folders, limit=6)
    print('\nSample run complete.')
