import os
import glob
import random
from shutil import copy2

import numpy as np
import cv2
import yaml
from ultralytics import YOLO


# Original dataset and new output dataset
ORIG_ROOT = "/deac/sta/classes/sta379/adena24/AboveWater_Drowning_Detection.v1i.yolov11"
ORIG_DATA_YAML = os.path.join(ORIG_ROOT, "data.yaml")

TURBID_ROOT = "/deac/sta/classes/sta379/adena24/AboveWater_Drowning_Detection_TurbidityOnly"

# Change "valid" -> "val" here if your dataset folder is named val
SPLITS = ["train", "valid", "test"]

PROJECT_NAME = "runs_phase2_ablation"
RUN_NAME = "yolo11n_turbidity_only"

# Optional preview settings
SAVE_PREVIEW = False
PREVIEW_DIR = os.path.join(TURBID_ROOT, "preview")
N_PREVIEWS = 5


def add_water_turbidity_haze(img, strength_range=(0.12, 0.32), particle_intensity=0.015, flip_vertical_prob=0.3):
    """Adds water-like haze (vertical gradient) + mild particle noise."""
    img_f = img.astype(np.float32) / 255.0

    alpha = random.uniform(*strength_range)

    # bluish haze in BGR
    base = np.array([0.90, 0.95, 1.00], dtype=np.float32)
    haze_color = np.clip(base + np.random.uniform(-0.04, 0.04, size=(3,)), 0.75, 1.0)

    h, w, _ = img_f.shape
    y = np.linspace(0, 1, h).reshape(-1, 1, 1)

    alpha_map = alpha * y
    if random.random() < flip_vertical_prob:
        alpha_map = alpha * (1.0 - y)

    alpha_map = np.repeat(alpha_map, 3, axis=2)
    hazy = img_f * (1 - alpha_map) + haze_color * alpha_map

    if particle_intensity > 0:
        hazy += np.random.normal(0.0, particle_intensity, size=img_f.shape).astype(np.float32)

    hazy = np.clip(hazy, 0, 1)
    hazy = cv2.GaussianBlur(hazy, (5, 5), 0)
    return (hazy * 255).astype(np.uint8)


def write_normalized_yaml(dst_root):
    """Copies original YAML metadata but points it to dst_root (absolute path)."""
    dst_yaml = os.path.join(dst_root, "data.yaml")

    with open(ORIG_DATA_YAML, "r") as f:
        data = yaml.safe_load(f)

    data["path"] = os.path.abspath(dst_root)
    data["train"] = "train/images"
    data["val"] = "valid/images"  # change to "val/images" if your folder is named val
    data["test"] = "test/images"

    with open(dst_yaml, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    print(f"data.yaml -> {dst_yaml}")


def build_turbidity_dataset():
    """Creates TURBID_ROOT where only TRAIN images are turbidity-augmented."""
    for split in SPLITS:
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(TURBID_ROOT, split, sub), exist_ok=True)

    if SAVE_PREVIEW:
        os.makedirs(PREVIEW_DIR, exist_ok=True)
        preview_count = 0

    for split in SPLITS:
        src_img_dir = os.path.join(ORIG_ROOT, split, "images")
        src_lbl_dir = os.path.join(ORIG_ROOT, split, "labels")
        dst_img_dir = os.path.join(TURBID_ROOT, split, "images")
        dst_lbl_dir = os.path.join(TURBID_ROOT, split, "labels")

        img_paths = sorted(glob.glob(os.path.join(src_img_dir, "*.*")))

        for img_path in img_paths:
            fname = os.path.basename(img_path)
            stem, _ = os.path.splitext(fname)

            src_lbl_path = os.path.join(src_lbl_dir, stem + ".txt")
            dst_lbl_path = os.path.join(dst_lbl_dir, stem + ".txt")
            dst_img_path = os.path.join(dst_img_dir, fname)

            if os.path.exists(src_lbl_path):
                copy2(src_lbl_path, dst_lbl_path)

            if split != "train":
                copy2(img_path, dst_img_path)
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue

            aug = add_water_turbidity_haze(img)
            cv2.imwrite(dst_img_path, aug)

            if SAVE_PREVIEW and preview_count < N_PREVIEWS:
                comp = np.hstack([img, aug])
                cv2.imwrite(os.path.join(PREVIEW_DIR, f"compare_{preview_count+1}.jpg"), comp)
                preview_count += 1

    write_normalized_yaml(TURBID_ROOT)

    if SAVE_PREVIEW:
        print(f"Previews -> {PREVIEW_DIR}")


def train_turbidity_model():
    """Trains YOLO11n on the turbidity-only dataset."""
    data_yaml = os.path.abspath(os.path.join(TURBID_ROOT, "data.yaml"))
    model = YOLO("yolo11n.pt")

    return model.train(
        data=data_yaml,
        epochs=50,
        imgsz=512,
        batch=4,
        workers=2,
        device="cpu",
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
        project=PROJECT_NAME,
        name=RUN_NAME,
    )


def main():
    build_turbidity_dataset()
    train_turbidity_model()


if __name__ == "__main__":
    main()
