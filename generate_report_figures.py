"""
Generate professional figures for the report:
- Figure 3.2: IOU Calculation Example (showing bounding box intersection)
- Figure 3.3: CLAHE Enhancement Example (showing before/after)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch
import os
from pathlib import Path
import urllib.request

# Create output directory
REPORT_FIGURES_DIR = Path("outputs/report_images")
REPORT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IOU) of two boxes."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
    
    # Calculate union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area if union_area > 0 else 0
    return iou, inter_area, union_area


def create_iou_visualization():
    """Create Figure 3.2: IOU Calculation Example with bounding boxes."""
    
    # Create a sample image (simulating a traffic/helmet detection scenario)
    img_width, img_height = 600, 400
    img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 240  # Light gray background
    
    # Add some texture to make it look like a road scene
    noise = np.random.randint(0, 20, (img_height, img_width, 3), dtype=np.uint8)
    img = np.clip(img.astype(int) + noise.astype(int), 0, 255).astype(np.uint8)
    
    # Add a simple road/background elements
    cv2.rectangle(img, (0, 250), (600, 400), (100, 100, 100), -1)  # Road
    cv2.circle(img, (150, 120), 40, (0, 165, 255), 3)  # Person/helmet
    
    # Define ground truth box and predicted box
    ground_truth_box = (80, 80, 220, 200)      # x_min, y_min, x_max, y_max
    predicted_box = (120, 100, 260, 220)
    
    # Calculate IOU
    iou, inter_area, union_area = calculate_iou(ground_truth_box, predicted_box)
    
    # Create figure with matplotlib
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Display the background image
    ax.imshow(img)
    
    # Draw ground truth box (green)
    gt_x1, gt_y1, gt_x2, gt_y2 = ground_truth_box
    gt_rect = Rectangle((gt_x1, gt_y1), gt_x2-gt_x1, gt_y2-gt_y1, 
                         linewidth=3, edgecolor='green', facecolor='none', 
                         label='Ground Truth', linestyle='-')
    ax.add_patch(gt_rect)
    ax.text(gt_x1, gt_y1-10, 'Ground Truth', color='green', fontsize=12, 
            fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Draw predicted box (red)
    pred_x1, pred_y1, pred_x2, pred_y2 = predicted_box
    pred_rect = Rectangle((pred_x1, pred_y1), pred_x2-pred_x1, pred_y2-pred_y1, 
                          linewidth=3, edgecolor='red', facecolor='none', 
                          label='Predicted', linestyle='--')
    ax.add_patch(pred_rect)
    ax.text(pred_x2+5, pred_y1, 'Predicted', color='red', fontsize=12, 
            fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Draw intersection box (blue)
    inter_x1 = max(gt_x1, pred_x1)
    inter_y1 = max(gt_y1, pred_y1)
    inter_x2 = min(gt_x2, pred_x2)
    inter_y2 = min(gt_y2, pred_y2)
    
    if inter_x2 > inter_x1 and inter_y2 > inter_y1:
        inter_rect = Rectangle((inter_x1, inter_y1), inter_x2-inter_x1, inter_y2-inter_y1,
                               linewidth=2, edgecolor='blue', facecolor='blue', 
                               alpha=0.3, label='Intersection')
        ax.add_patch(inter_rect)
    
    # Add metrics text
    metrics_text = f"""
    IOU Calculation Formula:
    IOU = Intersection Area / Union Area
    
    Intersection Area: {inter_area:.0f} pixels
    Union Area: {union_area:.0f} pixels
    IOU Score: {iou:.3f}
    
    Accuracy: {iou*100:.1f}%
    """
    
    ax.text(350, 150, metrics_text, fontsize=11, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, pad=1),
            verticalalignment='center')
    
    ax.set_xlim(0, img_width)
    ax.set_ylim(img_height, 0)
    ax.set_aspect('equal')
    ax.set_title('Figure 3.2: IOU (Intersection over Union) Calculation Example', 
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=11)
    ax.axis('off')
    
    plt.tight_layout()
    output_path = REPORT_FIGURES_DIR / "fig_3_2_iou_calculation.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()
    
    return str(output_path)


def create_clahe_visualization():
    """Create Figure 3.3: CLAHE Enhancement Example (before/after)."""
    
    # Create a sample dark/low contrast image (simulating night detection)
    img_height, img_width = 400, 600
    
    # Create a base image with low contrast
    base_img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 80  # Dark gray
    
    # Add some features (simulating vehicle/helmet)
    cv2.circle(base_img, (150, 120), 50, (90, 90, 90), -1)
    cv2.rectangle(base_img, (250, 100), (450, 250), (100, 100, 100), -1)
    cv2.circle(base_img, (120, 280), 40, (95, 95, 95), -1)
    
    # Add some noise to make it realistic
    noise = np.random.randint(-15, 15, base_img.shape, dtype=np.int16)
    original_img = np.clip(base_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Convert to LAB color space for CLAHE
    lab = cv2.cvtColor(original_img, cv2.COLOR_BGR2LAB)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_clahe = lab.copy()
    lab_clahe[:, :, 0] = clahe.apply(lab[:, :, 0])
    
    # Convert back to BGR
    enhanced_img = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    # Create side-by-side comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Original image
    axes[0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image\n(Low Contrast)', fontsize=12, fontweight='bold', pad=10)
    axes[0].axis('off')
    
    # Add statistics for original
    original_mean = original_img.mean()
    original_std = original_img.std()
    axes[0].text(0.5, -0.1, f'Mean Brightness: {original_mean:.1f}\nContrast (Std): {original_std:.1f}',
                transform=axes[0].transAxes, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    # Enhanced image
    axes[1].imshow(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title('After CLAHE Enhancement\n(Enhanced Contrast)', fontsize=12, fontweight='bold', pad=10)
    axes[1].axis('off')
    
    # Add statistics for enhanced
    enhanced_mean = enhanced_img.mean()
    enhanced_std = enhanced_img.std()
    axes[1].text(0.5, -0.1, f'Mean Brightness: {enhanced_mean:.1f}\nContrast (Std): {enhanced_std:.1f}',
                transform=axes[1].transAxes, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    fig.suptitle('Figure 3.3: CLAHE (Contrast Limited Adaptive Histogram Equalization) Enhancement Example',
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    output_path = REPORT_FIGURES_DIR / "fig_3_3_clahe_enhancement.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()
    
    return str(output_path)


def create_detection_comparison():
    """Bonus: Create a detection quality comparison showing IOU impact."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    scenarios = [
        {"name": "High IOU (Perfect Detection)", "iou": 0.95, "color": "green", "position": (0, 0)},
        {"name": "Good IOU (Acceptable)", "iou": 0.75, "color": "yellow", "position": (0, 1)},
        {"name": "Fair IOU (Marginal)", "iou": 0.50, "color": "orange", "position": (1, 0)},
        {"name": "Low IOU (Poor Detection)", "iou": 0.25, "color": "red", "position": (1, 1)},
    ]
    
    for scenario in scenarios:
        row, col = scenario["position"]
        ax = axes[row, col]
        
        # Create image
        img = np.ones((300, 400, 3), dtype=np.uint8) * 200
        cv2.rectangle(img, (50, 50), (200, 250), (180, 180, 180), -1)
        
        # Ground truth (fixed)
        gt_box = [80, 80, 180, 220]
        # Predicted (varies by IOU)
        offset = int((1 - scenario["iou"]) * 30)
        pred_box = [80 + offset, 80 + offset, 180 + offset, 220 + offset]
        
        iou, _, _ = calculate_iou(gt_box, pred_box)
        
        ax.imshow(img)
        
        # Draw boxes
        gt_rect = Rectangle((gt_box[0], gt_box[1]), gt_box[2]-gt_box[0], gt_box[3]-gt_box[1],
                           linewidth=2.5, edgecolor='green', facecolor='none')
        pred_rect = Rectangle((pred_box[0], pred_box[1]), pred_box[2]-pred_box[0], pred_box[3]-pred_box[1],
                             linewidth=2.5, edgecolor='red', facecolor='none', linestyle='--')
        ax.add_patch(gt_rect)
        ax.add_patch(pred_rect)
        
        # Add legend and IOU
        title_text = f"{scenario['name']}\nIOU: {iou:.2f} ({iou*100:.0f}%)"
        ax.set_title(title_text, fontsize=11, fontweight='bold', 
                    bbox=dict(boxstyle='round', facecolor=scenario['color'], alpha=0.6))
        ax.axis('off')
    
    fig.suptitle('Figure 3.4: Detection Quality by IOU Score',
                fontsize=14, fontweight='bold', y=0.98)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', lw=2.5, label='Ground Truth'),
        Line2D([0], [0], color='red', lw=2.5, linestyle='--', label='Predicted')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=10, 
              bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout()
    output_path = REPORT_FIGURES_DIR / "fig_3_4_detection_quality_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()
    
    return str(output_path)


def main():
    """Generate all report figures."""
    print("=" * 60)
    print("GENERATING PROFESSIONAL REPORT FIGURES")
    print("=" * 60)
    
    try:
        print("\n[1/3] Creating IOU Calculation Figure...")
        iou_path = create_iou_visualization()
        
        print("\n[2/3] Creating CLAHE Enhancement Figure...")
        clahe_path = create_clahe_visualization()
        
        print("\n[3/3] Creating Detection Quality Comparison...")
        comp_path = create_detection_comparison()
        
        print("\n" + "=" * 60)
        print("✓ ALL FIGURES GENERATED SUCCESSFULLY!")
        print("=" * 60)
        print(f"\nOutput Directory: {REPORT_FIGURES_DIR}")
        print(f"\nGenerated Files:")
        print(f"  1. fig_3_2_iou_calculation.png")
        print(f"  2. fig_3_3_clahe_enhancement.png")
        print(f"  3. fig_3_4_detection_quality_comparison.png")
        print("\nYou can now paste these images into your report!")
        
    except Exception as e:
        print(f"\n✗ Error generating figures: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
