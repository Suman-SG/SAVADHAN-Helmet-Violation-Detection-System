"""
Test Image Enhancement and Generate Report Figures
====================================================
Downloads sample images from URLs and creates professional comparison figures
Generates CLAHE enhancement and IOU visualization for the report
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import urllib.request
from pathlib import Path
import os

# Setup paths
BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "images"
REPORT_FIGURES_DIR = BASE_DIR / "outputs" / "report_images"
REPORT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# URLs for sample images
SAMPLE_URLS = {
    "pexels_traffic": "https://images.pexels.com/photos/30012213/pexels-photo-30012213.jpeg?cs=srgb&dl=pexels-sonu-krishna-2148280164-30012213.jpg&fm=jpg",
    "google_traffic": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTaT-QDI2ep8EqxrzE8cGfVQwBLqa6wO9aUAg&s"
}


def download_sample_images():
    """Download sample traffic images from URLs."""
    print("\n📥 Downloading sample images from URLs...")
    downloaded = []
    
    for name, url in SAMPLE_URLS.items():
        try:
            output_path = IMAGES_DIR / f"sample_{name}.jpg"
            print(f"  • Downloading {name}...", end=" ")
            urllib.request.urlretrieve(url, output_path, timeout=10)
            print(f"✓ Saved")
            downloaded.append(output_path)
        except Exception as e:
            print(f"✗ Failed: {str(e)[:50]}")
    
    return downloaded


def apply_clahe_enhancement(image_path):
    """Apply CLAHE enhancement and return before/after comparison."""
    print(f"\n🔍 Applying CLAHE enhancement to: {image_path.name}")
    
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"✗ Error: Could not load image")
        return None
    
    # Resize if too large (for display)
    if img.shape[1] > 800:
        scale = 800 / img.shape[1]
        img = cv2.resize(img, None, fx=scale, fy=scale)
    
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    lab_clahe = lab.copy()
    lab_clahe[:, :, 0] = clahe.apply(lab[:, :, 0])
    
    # Convert back to BGR
    enhanced_img = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    print(f"  ✓ Enhancement complete")
    return {
        'original': img,
        'enhanced': enhanced_img,
        'filename': image_path.name
    }


def create_clahe_comparison_figure(enhancement_data):
    """Create side-by-side CLAHE comparison figure."""
    
    original_img = enhancement_data['original']
    enhanced_img = enhancement_data['enhanced']
    
    # Calculate statistics
    original_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    enhanced_gray = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
    
    original_mean = original_gray.mean()
    original_std = original_gray.std()
    enhanced_mean = enhanced_gray.mean()
    enhanced_std = enhanced_gray.std()
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Original image
    axes[0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image\n(Original Contrast)', fontsize=13, fontweight='bold', 
                     bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    axes[0].axis('off')
    axes[0].text(0.5, -0.08, f'Brightness: {original_mean:.1f} | Contrast (Std): {original_std:.1f}',
                transform=axes[0].transAxes, ha='center', fontsize=10, family='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Enhanced image
    axes[1].imshow(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title('CLAHE Enhanced Image\n(Improved Contrast)', fontsize=13, fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    axes[1].axis('off')
    axes[1].text(0.5, -0.08, f'Brightness: {enhanced_mean:.1f} | Contrast (Std): {enhanced_std:.1f}',
                transform=axes[1].transAxes, ha='center', fontsize=10, family='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.suptitle('Figure 3.3: CLAHE (Contrast Limited Adaptive Histogram Equalization) Enhancement',
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    return fig


def calc_iou(box1, box2):
    """Calculate Intersection over Union."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


def create_iou_explanation_figure():
    """Create IOU explanation figure with visual examples."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    scenarios = [
        {"name": "Perfect Detection", "gt": (80, 60, 220, 180), "pred": (85, 65, 215, 175), "pos": (0, 0), "color": "lightgreen"},
        {"name": "Good Detection", "gt": (80, 60, 220, 180), "pred": (100, 75, 200, 170), "pos": (0, 1), "color": "lightyellow"},
        {"name": "Fair Detection", "gt": (80, 60, 220, 180), "pred": (120, 90, 210, 170), "pos": (1, 0), "color": "lightsalmon"},
        {"name": "Poor Detection", "gt": (80, 60, 220, 180), "pred": (150, 120, 240, 190), "pos": (1, 1), "color": "lightcoral"},
    ]
    
    for scenario in scenarios:
        row, col = scenario["pos"]
        ax = axes[row, col]
        
        # Create background image
        img = np.ones((300, 400, 3), dtype=np.uint8) * 220
        cv2.circle(img, (150, 120), 60, (100, 100, 150), -1)
        ax.imshow(img)
        
        # Calculate IOU
        iou = calc_iou(scenario["gt"], scenario["pred"])
        
        # Draw ground truth box (green)
        gt_box = scenario["gt"]
        gt_rect = Rectangle((gt_box[0], gt_box[1]), gt_box[2]-gt_box[0], gt_box[3]-gt_box[1], 
                            linewidth=2.5, edgecolor='green', facecolor='none')
        ax.add_patch(gt_rect)
        
        # Draw predicted box (red, dashed)
        pred_box = scenario["pred"]
        pred_rect = Rectangle((pred_box[0], pred_box[1]), pred_box[2]-pred_box[0], pred_box[3]-pred_box[1],
                              linewidth=2.5, edgecolor='red', linestyle='--', facecolor='none')
        ax.add_patch(pred_rect)
        
        ax.set_title(f'{scenario["name"]}\n(IOU = {iou:.3f})', fontsize=12, fontweight='bold', 
                    bbox=dict(boxstyle='round', facecolor=scenario["color"], alpha=0.8))
        ax.axis('off')
    
    # Add legend
    legend_elements = [
        Line2D([0], [0], color='green', lw=2.5, label='Ground Truth'),
        Line2D([0], [0], color='red', lw=2.5, linestyle='--', label='Predicted')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=12,
              bbox_to_anchor=(0.5, -0.02))
    
    fig.suptitle('Figure 3.2: IOU (Intersection over Union) - Detection Quality Assessment', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def main():
    """Main execution."""
    print("\n" + "="*70)
    print("IMAGE ENHANCEMENT & REPORT FIGURE GENERATION")
    print("="*70)
    
    # Step 1: Download sample images
    print("\n[STEP 1] Downloading sample images...")
    try:
        downloaded = download_sample_images()
        if downloaded:
            print(f"✓ Downloaded {len(downloaded)} sample images")
    except Exception as e:
        print(f"⚠ Could not download samples: {str(e)[:50]}")
        downloaded = []
    
    # Step 2: Create IOU explanation figure (standalone)
    print("\n[STEP 2] Creating IOU explanation figure...")
    try:
        fig = create_iou_explanation_figure()
        output_path = REPORT_FIGURES_DIR / "fig_3_2_iou_explanation.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: fig_3_2_iou_explanation.png")
        plt.close()
    except Exception as e:
        print(f"✗ Error: {str(e)[:100]}")
    
    # Step 3: Apply CLAHE to available images
    print("\n[STEP 3] Applying CLAHE enhancement to images...")
    
    # Test on thumb.jpg if it exists
    thumb_path = IMAGES_DIR / "thumb.jpg"
    if thumb_path.exists():
        enhancement_data = apply_clahe_enhancement(thumb_path)
        if enhancement_data:
            try:
                fig = create_clahe_comparison_figure(enhancement_data)
                output_path = REPORT_FIGURES_DIR / "fig_3_3_clahe_thumb.png"
                plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
                print(f"✓ Saved: fig_3_3_clahe_thumb.png")
                plt.close()
            except Exception as e:
                print(f"✗ Error: {str(e)[:100]}")
    
    # Apply to all other JPG files
    for img_path in IMAGES_DIR.glob("*.jpg"):
        if img_path.name != "thumb.jpg":
            enhancement_data = apply_clahe_enhancement(img_path)
            if enhancement_data:
                try:
                    fig = create_clahe_comparison_figure(enhancement_data)
                    output_filename = f"fig_3_3_clahe_{img_path.stem}.png"
                    output_path = REPORT_FIGURES_DIR / output_filename
                    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
                    print(f"✓ Saved: {output_filename}")
                    plt.close()
                except Exception as e:
                    print(f"✗ Error: {str(e)[:100]}")
    
    # Summary
    print("\n" + "="*70)
    print("✓ REPORT FIGURE GENERATION COMPLETE!")
    print("="*70)
    print(f"\nOutput Directory: {REPORT_FIGURES_DIR}")
    print("\n📊 Generated figures:")
    
    # List generated files
    figures = list(REPORT_FIGURES_DIR.glob("fig_*.png"))
    for i, fig_file in enumerate(sorted(figures), 1):
        print(f"  {i}. {fig_file.name}")
    
    print("\n✅ Ready to paste into your report!")


if __name__ == "__main__":
    main()
