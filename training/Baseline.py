import os
import glob
import random
from collections import Counter

import yaml
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False


DATASET_DIR = "/deac/sta/classes/sta379/adena24/AboveWater_Drowning_Detection.v1i.yolov11"
DATA_YAML = os.path.join(DATASET_DIR, "data.yaml")
SPLITS = ["train", "valid", "test"]

EPOCHS = 50
IMGSZ = 512
BATCH = 4
WORKERS = 2
DEVICE = "cpu"

SHOW_SAMPLES = True
N_SAMPLES = 3


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def split_counts(split_dir):
    imgs = glob.glob(os.path.join(split_dir, "images", "*"))
    lbls = glob.glob(os.path.join(split_dir, "labels", "*.txt"))
    return len(imgs), len(lbls)


def label_lines(dataset_dir, splits):
    for split in splits:
        files = glob.glob(os.path.join(dataset_dir, split, "labels", "*.txt"))
        for lf in files:
            with open(lf, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line


def class_counts(dataset_dir, splits):
    c = Counter()
    for line in label_lines(dataset_dir, splits):
        c[int(line.split()[0])] += 1
    return c


def box_stats(dataset_dir, splits):
    ws, hs = [], []
    for line in label_lines(dataset_dir, splits):
        p = line.split()
        if len(p) != 5:
            continue
        _, _, _, w, h = map(float, p)
        ws.append(w)
        hs.append(h)
    return ws, hs


def empty_label_files(dataset_dir, splits):
    out = []
    for split in splits:
        files = glob.glob(os.path.join(dataset_dir, split, "labels", "*.txt"))
        for lf in files:
            if os.path.getsize(lf) == 0:
                out.append(lf)
    return out


def plot_counts(counts, names):
    keys = sorted(counts.keys())
    labels = [names[i] if i < len(names) else str(i) for i in keys]
    values = [counts[i] for i in keys]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.title("Object count per class")
    plt.grid(axis="y", alpha=0.3)
    for i, v in enumerate(values):
        plt.text(i, v, str(v), ha="center", va="bottom")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 7))
    plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=140)
    plt.title("Class proportion")
    plt.tight_layout()
    plt.show()


def show_boxes(img_path, lbl_path):
    if not HAS_CV2:
        return

    img = cv2.imread(img_path)
    if img is None:
        return

    H, W = img.shape[:2]
    if os.path.exists(lbl_path):
        with open(lbl_path, "r") as f:
            for line in f:
                p = line.strip().split()
                if len(p) != 5:
                    continue
                _, x, y, w, h = map(float, p)
                x1 = int((x - w / 2) * W)
                y1 = int((y - h / 2) * H)
                x2 = int((x + w / 2) * W)
                y2 = int((y + h / 2) * H)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def clean_results_csv(run_dir):
    import pandas as pd

    p = os.path.join(run_dir, "results.csv")
    if not os.path.exists(p):
        return None

    df = pd.read_csv(p)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(how="any")
    df.to_csv(p, index=False)
    return p


def plot_training(results_csv):
    import pandas as pd

    if not results_csv or not os.path.exists(results_csv):
        return

    df = pd.read_csv(results_csv)
    x = df["epoch"] if "epoch" in df.columns else np.arange(len(df))

    plt.figure(figsize=(12, 6))
    for col in ["train/box_loss", "train/cls_loss", "metrics/mAP50(B)", "metrics/precision(B)", "metrics/recall(B)"]:
        if col in df.columns:
            plt.plot(x, df[col], label=col)
    plt.title("Training metrics")
    plt.xlabel("Epoch")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    cfg = load_yaml(DATA_YAML)
    names = cfg.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys())]

    for split in SPLITS:
        n_img, n_lbl = split_counts(os.path.join(DATASET_DIR, split))
        print(f"{split}: {n_img} images, {n_lbl} label files")

    counts = class_counts(DATASET_DIR, SPLITS)
    print("Class counts:", dict(counts))

    ws, hs = box_stats(DATASET_DIR, SPLITS)
    if ws:
        print(
            f"Box width:  mean={np.mean(ws):.4f} min={np.min(ws):.4f} max={np.max(ws):.4f} | "
            f"height: mean={np.mean(hs):.4f} min={np.min(hs):.4f} max={np.max(hs):.4f}"
        )

    empty = empty_label_files(DATASET_DIR, SPLITS)
    print("Empty label files:", len(empty))

    plot_counts(counts, names)

    if SHOW_SAMPLES and HAS_CV2:
        imgs = glob.glob(os.path.join(DATASET_DIR, "train/images", "*"))
        random.shuffle(imgs)
        for p in imgs[:N_SAMPLES]:
            lp = p.replace("/images/", "/labels/")
            lp = os.path.splitext(lp)[0] + ".txt"
            show_boxes(p, lp)

    model = YOLO("yolo11n.pt")
    res = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=WORKERS,
        device=DEVICE,
        mosaic=0.5,
        mixup=0.0,
        copy_paste=0.0,
        erasing=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.05,
        scale=0.3,
        fliplr=0.5,
        box=7.5,
        cls=1.2,
        dfl=1.5,
        close_mosaic=10,
    )

    run_dir = str(getattr(res, "save_dir", "")) if hasattr(res, "save_dir") else ""
    if run_dir:
        results_csv = clean_results_csv(run_dir)
        plot_training(results_csv)

        best_pt = os.path.join(run_dir, "weights", "best.pt")
        if os.path.exists(best_pt):
            YOLO(best_pt).val(data=DATA_YAML, save=True, plots=True)


if __name__ == "__main__":
    main()
