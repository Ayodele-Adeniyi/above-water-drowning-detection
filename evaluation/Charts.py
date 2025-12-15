import matplotlib.pyplot as plt
import numpy as np

# Metrics you already extracted
models = ["Baseline", "Turbidity", "Color", "Glare", "Occlusion", "Phase3", "Ensemble"]

# YOLO models have mAP@50 here; ensemble doesn't in your summary
map50 = [0.7676, 0.7438, 0.7472, 0.7472, 0.6811, 0.7488, None]

# Drowning-class metrics (YOLO drowning row + ensemble eval)
recall_drowning = [0.711, 0.692, 0.685, 0.685, 0.546, 0.703, 0.8059]
precision_drowning = [0.759, 0.791, 0.706, 0.706, 0.654, 0.780, 0.6094]


def chart1_map_vs_recall():
    """Bar chart: YOLO mAP@50 vs drowning recall; ensemble shows recall only."""
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(models))
    width = 0.35

    ax.bar(x[:-1] - width / 2, map50[:-1], width, label="mAP@50 (YOLO)", alpha=0.85)
    ax.bar(x[:-1] + width / 2, recall_drowning[:-1], width, label="Recall (Drowning)", alpha=0.85)
    ax.bar(x[-1], recall_drowning[-1], width * 2, label="Ensemble Recall (Drowning)", alpha=0.90)

    ax.set_ylabel("Score")
    ax.set_title("Model Comparison: mAP@50 vs. Drowning Recall")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylim(0.50, 0.90)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig("chart1_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()


def chart2_precision_recall_scatter():
    """Scatter plot: drowning precision vs recall for all models."""
    fig, ax = plt.subplots(figsize=(10, 8))

    for m, r, p in zip(models, recall_drowning, precision_drowning):
        ax.scatter(r, p, s=240, alpha=0.8, edgecolors="white", linewidth=2)
        ax.annotate(m, (r, p), xytext=(6, 6), textcoords="offset points", fontsize=9)

    ax.set_xlabel("Recall (Drowning)")
    ax.set_ylabel("Precision (Drowning)")
    ax.set_title("Precision–Recall Tradeoff Across Models (Drowning Class)")
    ax.set_xlim(0.50, 0.90)
    ax.set_ylim(0.50, 0.90)
    ax.grid(True, alpha=0.3)

    # Baseline reference lines
    ax.axhline(y=precision_drowning[0], linestyle="--", alpha=0.5, linewidth=1)
    ax.axvline(x=recall_drowning[0], linestyle="--", alpha=0.5, linewidth=1)

    plt.tight_layout()
    plt.savefig("chart2_precision_recall.png", dpi=150, bbox_inches="tight")
    plt.show()


def chart3_phase2_ablation():
    """Bar chart: Phase 2 ablations (mAP@50 and drowning recall)."""
    distortions = ["Turbidity", "Color", "Glare", "Occlusion", "Phase3\n(Combined)"]
    distortion_map50 = [0.7438, 0.7472, 0.7472, 0.6811, 0.7488]
    distortion_recall = [0.692, 0.685, 0.685, 0.546, 0.703]

    x = np.arange(len(distortions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, distortion_map50, width, label="mAP@50", alpha=0.85)
    bars2 = ax.bar(x + width / 2, distortion_recall, width, label="Recall (Drowning)", alpha=0.85)

    ax.set_ylabel("Score")
    ax.set_title("Per-Distortion Model Performance (Phase 2 Ablation)")
    ax.set_xticks(x)
    ax.set_xticklabels(distortions)
    ax.set_ylim(0.50, 0.85)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    # Small value labels (keep it simple)
    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig("chart3_per_distortion.png", dpi=150, bbox_inches="tight")
    plt.show()


def chart5_safety_first_baseline_vs_ensemble():
    """Baseline vs Ensemble: recall gain vs precision drop (drowning class)."""
    base_r, ens_r = recall_drowning[0], recall_drowning[-1]
    base_p, ens_p = precision_drowning[0], precision_drowning[-1]

    labels = ["Baseline", "Ensemble"]
    rec = [base_r, ens_r]
    prec = [base_p, ens_p]

    abs_gain = ens_r - base_r
    rel_gain = (abs_gain / base_r) * 100.0

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, rec, width, label="Recall", alpha=0.85)
    bars2 = ax.bar(x + width / 2, prec, width, label="Precision", alpha=0.85)

    ax.set_ylabel("Score")
    ax.set_title("Recall vs Precision: Safety-First Design Choice (Drowning Class)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0.50, 0.90)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=11)

    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Show recall improvement clearly
    ax.annotate(
        "",
        xy=(0.15, ens_r),
        xytext=(0.15, base_r),
        arrowprops=dict(arrowstyle="->", lw=2),
    )
    ax.text(
        0.30,
        (base_r + ens_r) / 2,
        f"+{abs_gain:.3f} abs\n(+{rel_gain:.1f}% rel)",
        fontsize=10,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig("chart5_recall_precision_improvement.png", dpi=150, bbox_inches="tight")
    plt.show()


def main():
    # Run whichever ones you need
    chart1_map_vs_recall()
    chart2_precision_recall_scatter()
    chart3_phase2_ablation()
    chart5_safety_first_baseline_vs_ensemble()


if __name__ == "__main__":
    main()
