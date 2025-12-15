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

COLOR_ROOT = "/deac/sta/classes/sta379/adena24/AboveWater_Drowning_Detection_ColorOnly"

# Change "valid" -> "val" here if your dataset folder is named val
SPLITS = ["train", "valid", "test"]

PROJECT_NAME = "runs_phase2_ablation"
RUN_NAME = "yolo11n_color_only"


def add_underwater_color_distortion(
    img,
    red_atten_range=(0.85, 1.0),
    blue_boost_range=(0.5, 1.0),
    green_boost_range=(0.4, 0.8),
    fog_strength_range=(0.3, 0.6),
    darkness_range=(0.2, 0.5),
):
    """
    Strong underwater-ish color shift:
    - red fades with "depth"
    - blue/green boost with "depth"
    - teal fog/haze increases toward the bottom
    - slight darkening toward the bottom
    """
    img_f = img.astype(np.float32) / 255.0
    h, w, _ = img_f.shape

    # vertical gradient: 0 at top, 1 at bottom
    y = np.linspace(0, 1, h).reshape(-1, 1)

    red_att = random.uniform(*red_atten_range)
    blue_boost = random.uniform(*blue_boost_range)
    green_boost = random.uniform(*green_boost_range)
    fog_strength = random.uniform(*fog_strength_range)
    darkness = random.uniform(*darkness_range)

    # OpenCV is BGR
    B = img_f[:, :, 0]
    G = img_f[:, :, 1]
    R = img_f[:, :, 2]

    # Red attenuation
    red_scale = 1.0 - red_att * y
    red_scale = np.repeat(red_scale, w, axis=1)
    R = R * red_scale

    # Blue/green boost
    blue_scale = 1.0 + blue_boost * y
    green_scale = 1.0 + green_boost * y
    blue_scale = np.repeat(blue_scale, w, axis=1)
    green_scale = np.repeat(green_scale, w, axis=1)

    B = np.clip(B * blue_scale, 0, 1)
    G = np.clip(G * green_scale, 0, 1)

    merged = np.stack([B, G, R], axis=2)

    # Teal-ish fog (stronger at the bottom)
    fog_color = np.array([0.8, 1.0, 1.0], dtype=np.float32)  # BGR
    fog_map = (fog_strength * y).reshape(h, 1, 1)
    fog_map = np.repeat(fog_map, w, axis=1)
    fog_map = np.repeat(fog_map, 3, axis=2)

    fogged = (1 - fog_map) * merged + fog_map * fog_color

    # Darken with depth
    dark_map = 1.0 - darkness * y
    dark_map = np.repeat(dark_map, w, axis=1)
    fogged[:, :, 0] *= dark_map
    fogged[:, :, 1] *= dark_map
    fogged[:, :, 2] *= dark_map

    out = cv2.GaussianBlur(fogged, (9, 9), 0)
    out = np.clip(out, 0, 1)

    return (out * 255).astype(np.uint8)


def build_color_dataset():
    """Creates COLOR_ROOT where only TRAIN images are color-distorted."""
    for split in SPLITS:
        for sub in ["images", "labels"]:
            os.makedirs(os.path.join(COLOR_ROOT, split, sub), exist_ok=True)

    for split in SPLITS:
        src_img_dir = os.path.join(ORIG_ROOT, split, "images")
        src_lbl_dir = os.path.join(ORIG_ROOT, split, "labels")
        dst_img_dir = os.path.join(COLOR_ROOT, split, "images")
        dst_lbl_dir = os.path.join(COLOR_ROOT, split, "labels")

        img_paths = sorted(glob.glob(os.path.join(src_img_dir, "*.*")))

        for img_path in img_paths:
            fname = os.path.basename(img_path)
            stem, _ = os.path.splitext(fname)

            src_lbl_path = os.path.join(src_lbl_dir, stem + ".txt")
            dst_lbl_path = os.path.join(dst_lbl_dir, stem + ".txt")
            dst_img_path = os.path.join(dst_img_dir, fname)

            # labels are unchanged
            if os.path.exists(src_lbl_path):
                copy2(src_lbl_path, dst_lbl_path)

            # only train is augmented
            if split == "train":
                img = cv2.imread(img_path)
                if img is None:
                    continue
                aug = add_underwater_color_distortion(img)
                cv2.imwrite(dst_img_path, aug)
            else:
                copy2(img_path, dst_img_path)

    # Rebuild data.yaml so Ultralytics points to this dataset root
    new_yaml_path = os.path.join(COLOR_ROOT, "data.yaml")
    with open(ORIG_DATA_YAML, "r") as f:
        data = yaml.safe_load(f)

    data["path"] = os.path.abspath(COLOR_ROOT)
    data["train"] = "train/images"
    data["val"] = "valid/images"   # change to "val/images" if your folder is named val
    data["test"] = "test/images"

    with open(new_yaml_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def preview_augmented_samples(n=5):
    """Writes side-by-side images to COLOR_ROOT/preview (original | augmented)."""
    original_folder = os.path.join(ORIG_ROOT, "train", "images")
    save_folder = os.path.join(COLOR_ROOT, "preview")
    os.makedirs(save_folder, exist_ok=True)

    img_paths = sorted(glob.glob(os.path.join(original_folder, "*")))
    for i, orig_path in enumerate(img_paths[:n]):
        orig = cv2.imread(orig_path)
        if orig is None:
            continue
        after = add_underwater_color_distortion(orig)
        combined = np.hstack([orig, after])
        cv2.imwrite(os.path.join(save_folder, f"compare_{i+1}.jpg"), combined)


def train_color_model():
    """Trains YOLO11n on the color-only dataset."""
    data_yaml = os.path.abspath(os.path.join(COLOR_ROOT, "data.yaml"))
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
    build_color_dataset()
    preview_augmented_samples(n=5)
    train_color_model()


if __name__ == "__main__":
    main()
