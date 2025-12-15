import pandas as pd
import matplotlib
matplotlib.use("Agg")  # use non-GUI backend (important on DEAC)
import matplotlib.pyplot as plt

# 🔹 Path to your results.csv
csv_path = "/deac/sta/classes/sta379a-sp-2025/adena24/runs/detect/train8/results.csv"

# Read and clean
df = pd.read_csv(csv_path)

# Sometimes YOLO logs can contain bad strings like '0z.61519'
for col in df.columns:
    if col != "epoch":
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna()  # drop any bad rows

epochs = df["epoch"]

# 1) Train losses
plt.figure(figsize=(8, 5))
plt.plot(epochs, df["train/box_loss"], label="box_loss")
plt.plot(epochs, df["train/cls_loss"], label="cls_loss")
plt.plot(epochs, df["train/dfl_loss"], label="dfl_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Losses over Epochs")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("train_losses.png")
plt.close()

# 2) Validation metrics: Precision, Recall, mAP50, mAP50-95
plt.figure(figsize=(8, 5))
plt.plot(epochs, df["metrics/precision(B)"], label="precision")
plt.plot(epochs, df["metrics/recall(B)"], label="recall")
plt.plot(epochs, df["metrics/mAP50(B)"], label="mAP50")
plt.plot(epochs, df["metrics/mAP50-95(B)"], label="mAP50-95")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Validation Metrics over Epochs")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("val_metrics.png")
plt.close()

print("Saved: train_losses.png and val_metrics.png")
