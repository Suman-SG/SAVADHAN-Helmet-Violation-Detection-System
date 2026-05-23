import os
import matplotlib.pyplot as plt


def make_chart(out_path):
    # Percentage-based metrics
    perc_metrics = {
        'Helmet Detection\nAccuracy': 100.0,
        'Plate Detection\nAccuracy': 60.0,
        'OCR Character\nAccuracy': 59.6,
        'Precision': 89.0,
        'Recall': 88.0,
    }

    other_metrics = {
        'Inference Speed\n(ms/image)': 40,
        'Duplicate Prevention\n(hours)': 12,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})

    # Top: percentages
    names = list(perc_metrics.keys())
    values = [perc_metrics[n] for n in names]
    bars = ax1.bar(names, values, color=['#2d7fb8', '#7fb82d', '#b82d7f', '#b88a2d', '#2db89a'])
    ax1.set_ylim(0, 110)
    ax1.set_ylabel('Percent (%)')
    ax1.set_title('Performance Metrics — Percent Values')
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    for b, v in zip(bars, values):
        ax1.annotate(f"{v:.1f}%" if isinstance(v, float) else f"{v}%",
                     xy=(b.get_x() + b.get_width() / 2, v), xytext=(0, 6),
                     textcoords='offset points', ha='center', va='bottom', fontsize=10)

    # Bottom: non-percent metrics as horizontal bars with labels
    other_names = list(other_metrics.keys())
    other_values = [other_metrics[n] for n in other_names]
    y_pos = range(len(other_names))
    ax2.barh(y_pos, other_values, color=['#f39c12', '#8e44ad'])
    ax2.set_yticks(list(y_pos))
    ax2.set_yticklabels(other_names)
    ax2.invert_yaxis()
    ax2.set_xlabel('Value')
    ax2.set_title('Other Metrics')
    for i, v in enumerate(other_values):
        ax2.annotate(f"{v}", xy=(v, i), xytext=(6, 0), textcoords='offset points', va='center')

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)


if __name__ == '__main__':
    out = 'outputs/report_images/performance_metrics_chart.png'
    make_chart(out)
    print(f'Saved: {out}')
