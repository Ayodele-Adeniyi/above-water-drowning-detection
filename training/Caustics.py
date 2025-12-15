import os
import glob
import random
from shutil import copy2

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


# Paths to original and glare-only datasets
ORIG_ROOT = "/deac/sta/classes/sta379/adena24/AboveWater_Drowning_Detection.v1i.yolov11"
GLARE_ROOT = "/deac/sta/classes/sta379/adena24/AboveWater_Drowning_Detection_GlareOnly"
ORIG_DATA_YAML = os.path.join(ORIG_ROOT, "data.yaml")

# Dataset splits and run info
SPLITS = ["train", "valid", "test"]
PROJECT_NAME = "runs_phase2_ablation"
RUN_NAME = "yolo11n_glare_only"


def add_glare_caustics(
    img,
    num_patches_range=(25, 60),
    intensity_range=(0.4, 0.9),
    top_bias=0.7,
):
    """
    Adds synthetic glare / caustics to an image.
    Glare is biased toward the top of the image to mimic surface reflections.
    """
    img = img.astype(np.float32) / 255.0
    h, w, _ = img.shape

    # Light map that will hold bright reflection blobs
    light_map = np.zeros((h, w), dtype=np.float32)
    num_patches = random.randint(*num_patches_range)

    for _ in range(num_patches):
        cx = random.randint(0, w - 1)
        cy = random.randint(0, max(1, int(h * top_bias)) - 1)

        rx = random.randint(int(0.02 * w), int(0.12 * w))
        ry = random.randint(int(0.01 * h), int(0.10 * h))

        cv2.ellipse(
            light_map,
            (cx, cy),
            (rx, ry),
            angle=random.uniform(0, 180),
            startAngle=0,
            endAngle=360,
            color=random.uniform(0.5, 1.0),
            thickness=-1,
        )

    # Blur to turn hard blobs into soft glare
    ksize = max(31, int(0.1 * min(h, w)))
    if ksize % 2 == 0:
        ksize += 1
    light_map = cv2.GaussianBlur(light_map, (ksize, ksize), 0)

    # Normalize and scale intensity
    max_val = light_map.max()
    if max_val > 0:
        light_map /= max_val
    light_map *= random.uniform(*intensity_range)

    # Slight warm tint (more red / green)
    light_map_3 = np.stack(
        [light_map * 0.9, light_map * 1.0, light_map * 1.1],
        axis=2,
    )

    out = np.clip(img + light_map_3, 0.0, 1.0)
    out = cv2.GaussianBlur(out, (3, 3), 0)

    return (out * 255).astype(np.uint8)


def build_glare_dataset():
    """
    Creates a new dataset:
    - Train images are glare-augmented
    - Valid and test images are copied unchanged
    - Labels are copied for all splits
    """
    for split in SPLITS:
        for sub in ["images", "labels"]:
            os.makedirs(os.path.join(GLARE_ROOT, split, sub), exist_ok=True)

    for split in SPLITS:
        src_img = os.path.join(ORIG_ROOT, split, "images")
        src_lbl = os.path.join(ORIG_ROOT, split, "labels")
        dst_img = os.path.join(GLARE_ROOT, split, "images")
        dst_lbl = os.path.join(GLARE_ROOT, split, "labels")

        for img_path in glob.glob(os.path.join(src_img, "*.*")):
            name = os.path.basename(img_path)
            stem = os.path.splitext(name)[0]

            lbl_src = os.path.join(src_lbl, stem + ".txt")
            lbl_dst = os.path.join(dst_lbl, stem + ".txt")

            if os.path.exists(lbl_src):
                copy2(lbl_src, lbl_dst)

            if split == "train":
                img = cv2.imread(img_path)
                if img is None:
                    continue
                cv2.imwrite(
                    os.path.join(dst_img, name),
                    add_glare_caustics(img),
                )
            else:
                copy2(img_path, os.path.join(dst_img, name))

    # Write a new data.yaml pointing to the glare dataset
    with open(ORIG_DATA_YAML, "r") as f:
        data = yaml.safe_load(f)

    data["path"] = "."
    data["train"] = "train/images"
    data["val"] = "valid/images"
    data["test"] = "test/images"

    with open(os.path.join(GLARE_ROOT, "data.yaml"), "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def preview_augmented_samples(n=5):
    """
    Saves side-by-side original vs augmented images
    to quickly verify the glare effect.
    """
    src = os.path.join(ORIG_ROOT, "train", "images")
    dst = os.path.join(GLARE_ROOT, "preview")
    os.makedirs(dst, exist_ok=True)

    for i, path in enumerate(sorted(glob.glob(os.path.join(src, "*")))[:n]):
        img = cv2.imread(path)
        if img is None:
            continue

        aug = add_glare_caustics(img)
        cv2.imwrite(
            os.path.join(dst, f"compare_{i+1}.jpg"),
            np.hstack([img, aug]),
        )


def train_glare_model():
    """
    Trains YOLOv11n on the glare-only dataset
    using the same hyperparameters as Phase 1.
    """
    model = YOLO("yolo11n.pt")
    return model.train(
        data=os.path.join(GLARE_ROOT, "data.yaml"),
        epochs=50,
        imgsz=512,
        batch=4,
        workers=2,
        device="cpu",
        mosaic=0.5,
        mixup=0.0,
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
    build_glare_dataset()
    preview_augmented_samples(n=5)

    if input("Proceed to training? [y/N]: ").strip().lower() == "y":
        train_glare_model()


if __name__ == "__main__":
    main()
