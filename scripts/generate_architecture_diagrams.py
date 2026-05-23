#!/usr/bin/env python3
"""Generate two system architecture diagrams (overview and detailed) for the helmet detection project.
Saves PNGs to `outputs/` directory.
"""
import os
import math
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ArrowStyle, ConnectionPatch

try:
    import seaborn as sns
    sns.set(style="whitegrid")
except Exception:
    plt.style.use('seaborn-white')


def make_box(ax, xy, w, h, text, boxstyle="round,pad=0.3", fc="#f7f7f7", ec="#333333"):
    x, y = xy
    box = FancyBboxPatch((x, y), w, h, boxstyle=boxstyle, linewidth=1.2, facecolor=fc, edgecolor=ec)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)
    return box


def draw_overview(ax):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # positions and sizes
    boxes = []
    boxes.append(((0.5, 3.0), 2.0, 1.0, 'Camera / CCTV'))
    boxes.append(((3.0, 3.0), 2.2, 1.0, 'Frame Sampling'))
    boxes.append(((5.0, 3.0), 2.6, 1.0, 'Preprocessing\n(CLAHE, Gamma)'))
    boxes.append(((8.0, 3.0), 2.0, 1.0, 'YOLOv8 Detector\n(rider, helmet, plate)'))
    boxes.append(((8.0, 1.5), 2.0, 1.0, 'Plate Crop'))
    boxes.append(((5.0, 1.5), 2.0, 1.0, 'SR (ESRGAN) +\nThresholding'))
    boxes.append(((2.5, 1.5), 3.0, 1.0, 'OCR (EasyOCR -> Tesseract fallback)'))
    boxes.append(((0.5, 0.2), 3.5, 1.0, 'DB / Evidence packer\nEmail / Dashboard'))

    drawn = [make_box(ax, pos, w, h, txt) for (pos, w, h, txt) in boxes]

    # arrows
    def connect(b1, b2, loc1=(1, 0.5), loc2=(0, 0.5)):
        x1 = b1.get_x() + b1.get_width() * loc1[0]
        y1 = b1.get_y() + b1.get_height() * loc1[1]
        x2 = b2.get_x() + b2.get_width() * loc2[0]
        y2 = b2.get_y() + b2.get_height() * loc2[1]
        con = ConnectionPatch((x1, y1), (x2, y2), "data", "data",
                              arrowstyle=ArrowStyle("->", head_length=6, head_width=4),
                              linewidth=1.0, color="#444444")
        ax.add_patch(con)

    connect(drawn[0], drawn[1])
    connect(drawn[1], drawn[2])
    connect(drawn[2], drawn[3])
    connect(drawn[3], drawn[4], loc1=(0.8, 0.5), loc2=(0.2, 0.5))
    connect(drawn[4], drawn[5], loc1=(0.0, 0.5), loc2=(1.0, 0.5))
    connect(drawn[5], drawn[6], loc1=(0.0, 0.5), loc2=(1.0, 0.5))
    connect(drawn[6], drawn[7], loc1=(0.0, 0.5), loc2=(0.5, 1.0))

    # add small labels for thresholds
    ax.text(7.0, 3.8, 'Conf. Thresh: Tc', fontsize=8, color='#222222')
    ax.text(6.8, 1.1, 'SR scale x2', fontsize=8, color='#222222')
    ax.set_title('System Overview: High-level Pipeline', fontsize=12, pad=12)


def draw_detail(ax):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Detector box with heads
    det = make_box(ax, (1.0, 5.0), 4.5, 2.0, 'YOLOv8 Detector\n(backbone -> heads)\nOutputs: rider, helmet, plate')

    # detection outputs
    rider = make_box(ax, (6.0, 6.2), 2.2, 1.0, 'Rider')
    helmet = make_box(ax, (6.0, 5.0), 2.2, 1.0, 'Helmet / No-Helmet')
    plate = make_box(ax, (6.0, 3.8), 2.2, 1.0, 'Plate (bbox)')

    # plate OCR pipeline
    crop = make_box(ax, (9.0, 3.8), 2.2, 1.0, 'Crop & Normalize')
    sr = make_box(ax, (9.0, 2.4), 2.2, 1.0, 'ESRGAN (Super-Resolution)')
    proc = make_box(ax, (6.0, 1.6), 2.2, 1.0, 'Thresholding / Sharpen')
    ocr = make_box(ax, (3.5, 1.6), 2.6, 1.0, 'Hybrid OCR\nEasyOCR -> Tesseract fallback')
    validate = make_box(ax, (1.0, 1.6), 2.2, 1.0, 'Plate Validation\nRegex + State code + Correction')

    # arrows
    def conn(a, b):
        x1 = a.get_x() + a.get_width()
        y1 = a.get_y() + a.get_height() / 2
        x2 = b.get_x()
        y2 = b.get_y() + b.get_height() / 2
        con = ConnectionPatch((x1, y1), (x2, y2), "data", "data",
                              arrowstyle=ArrowStyle("->", head_length=6, head_width=4),
                              linewidth=1.0, color="#333333")
        ax.add_patch(con)

    conn(det, rider)
    conn(det, helmet)
    conn(det, plate)
    conn(plate, crop)
    conn(crop, sr)
    conn(sr, proc)
    conn(proc, ocr)
    conn(ocr, validate)

    # add MR scoring box and DB
    mr = make_box(ax, (1.0, -0.2), 3.5, 1.0, 'MR Score\n(Detection & OCR confidence)')
    db = make_box(ax, (6.0, -0.2), 3.5, 1.0, 'DB / Evidence\nInvoice & Email')
    conn(validate, mr)
    conn(mr, db)

    ax.set_title('Detailed Diagram: Detector → Plate OCR → Validation', fontsize=12, pad=12)


def main():
    os.makedirs('outputs', exist_ok=True)

    # Overview diagram
    fig1, ax1 = plt.subplots(figsize=(11, 4.5))
    draw_overview(ax1)
    out1 = os.path.join('outputs', 'architecture_overview.png')
    fig1.savefig(out1, dpi=200, bbox_inches='tight')
    print('Saved', out1)

    # Detailed diagram
    fig2, ax2 = plt.subplots(figsize=(11, 6))
    draw_detail(ax2)
    out2 = os.path.join('outputs', 'architecture_detail.png')
    fig2.savefig(out2, dpi=200, bbox_inches='tight')
    print('Saved', out2)


if __name__ == '__main__':
    main()
