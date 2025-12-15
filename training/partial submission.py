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

OCCL_ROOT = "/deac/sta/classes/sta379/adena24/AboveWater_Drowning_Detection_OcclusionOnly_Realistic"

# Change "valid" -> "val" here if your dataset folder is named val
SPLITS = ["train", "valid", "test"]

PROJECT_NAME = "runs_phase2_ablation"
RUN_NAME = "yolo11n_occlusion_only_realistic"


def read_yolo_labels(label_path):
    """Read YOLO labels (cls cx cy w h). Returns [] if missing/empty."""
    if (not os.path.exists(label_path)) or os.path.getsize(label_path) == 0:
        return []

    labels = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, cx, cy, w, h = map(float, parts)
            labels.append((cls, cx, cy, w, h))
    return labels


def add_realistic_submersion(
    img,
    labels,
    submersion_frac_range=(0.25, 0.45),
    blur_strength_range=(3, 6),
    tint_strength=0.04,
    desaturate_strength=0.08,
    ripple_strength=1.2,
    feather_px=40,
):
    """
    Applies a soft "underwater" occlusion to the lower half of the first labeled person.
    Uses an organic mask (ellipse + noise), blur, tint, desaturation, and slight ripple warp.
    """
    img_f = img.astype(np.float32) / 255.0
    H, W, _ = img_f.shape

    if not labels:
        return img

    cls, cx, cy, w, h = labels[0]
    px_cx = int(cx * W)
    px_cy = int(cy * H)
    px_w = int(w * W)
    px_h = int(h * H)

    person_bottom = px_cy + px_h // 2
    if person_bottom < H * 0.4:
        return img

    sub_frac = random.uniform(*submersion_frac_range)
    sub_pixels = int(px_h * sub_frac)

    y0 = px_cy  # treat center y as "waist-ish" point
    y1 = min(H, px_cy + sub_pixels)
    if y1 <= y0:
        return img

    x0 = max(0, px_cx - int(px_w * 0.6))
    x1 = min(W, px_cx + int(px_w * 0.6))

    region = img_f[y0:y1, x0:x1]
    Rh, Rw, _ = region.shape
    if Rh <= 0 or Rw <= 0:
        return img

    # Mask: ellipse + blur + small noise so it doesn't look like a rectangle
    yy, xx = np.indices((Rh, Rw))
    cy_r = Rh * 0.2
    cx_r = Rw * 0.5
    ry = Rh * 0.85
    rx = Rw * 0.48

    ellipse = ((yy - cy_r) ** 2) / (ry ** 2) + ((xx - cx_r) ** 2) / (rx ** 2)
    mask_core = (ellipse < 1).astype(np.float32)
    mask_core = np.clip(mask_core + np.linspace(0, 1, Rh).reshape(Rh, 1) * 0.4, 0, 1)

    kf = feather_px | 1
    mask = cv2.GaussianBlur(mask_core, (kf, kf), 0)

    noise = cv2.GaussianBlur(
        np.random.uniform(0, 0.25, (Rh, Rw)).astype(np.float32),
        (kf, kf),
        0,
    )
    mask = np.clip(mask + noise * 0.2, 0, 1)

    # Blur inside the region (loss of detail)
    k = random.randint(*blur_strength_range)
    if k % 2 == 0:
        k += 1
    region_blur = cv2.GaussianBlur(region, (k, k), 0)

    # Slight desaturation
    gray = cv2.cvtColor((region_blur * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY) / 255.0
    gray3 = np.stack([gray, gray, gray], axis=2)
    region_desat = region_blur * (1 - desaturate_strength) + gray3 * desaturate_strength

    # Tint (green/blue underwater)
    tint = np.array([0.0, tint_strength, tint_strength]).reshape(1, 1, 3)
    region_tinted = np.clip(region_desat + tint, 0, 1)

    # Slight ripple warp (refraction effect)
    ry_grid, rx_grid = np.indices((Rh, Rw))
    ripple = (np.sin(rx_grid / 15) + np.cos(ry_grid / 22)) * ripple_strength
    yy2 = np.clip(ry_grid + ripple.astype(int), 0, Rh - 1)
    xx2 = np.clip(rx_grid + ripple.astype(int), 0, Rw - 1)
    region_warped = region_tinted[yy2, xx2]

    # Blend back into the original image
    mask3 = np.repeat(mask[:, :, None], 3, axis=2)
    out = img_f.copy()
    out[y0:y1, x0:x1] = region * (1 - mask3) + region_warped * mask3

    out = np.clip(out, 0.0, 1.0)
    return (out * 255).astype(np.uint8)


def build_occlusion_dataset():
    """Creates OCCL_ROOT where only TRAIN images are occluded using labels."""
    for split in SPLITS:
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(OCCL_ROOT, split, sub), exist_ok=True)

    for split in SPLITS:
        src_img_dir = os.path.join(ORIG_ROOT, split, "images")
        src_lbl_dir = os.path.join(ORIG_ROOT, split, "labels")
        dst_img_dir = os.path.join(OCCL_ROOT, split, "images")
        dst_lbl_dir = os.path.join(OCCL_ROOT, split, "labels")

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

            # valid/test are copied unchanged
            if split != "train":
                copy2(img_path, dst_img_path)
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue

            labels = read_yolo_labels(src_lbl_path)
            aug = add_realistic_submersion(img, labels)
            cv2.imwrite(dst_img_path, aug)

    # Rebuild data.yaml so Ultralytics points to this dataset root
    new_yaml_path = os.path.join(OCCL_ROOT, "data.yaml")
    with open(ORIG_DATA_YAML, "r") as f:
        data = yaml.safe_load(f)

    data["path"] = os.path.abspath(OCCL_ROOT)
    data["train"] = "train/images"
    data["val"] = "valid/images"  # change to "val/images" if your folder is named val
    data["test"] = "test/images"

    with open(new_yaml_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def preview_augmented_samples(n=5):
    """Writes side-by-side images to OCCL_ROOT/preview (original | occluded)."""
    original_folder = os.path.join(ORIG_ROOT, "train", "images")
    label_folder = os.path.join(ORIG_ROOT, "train", "labels")
    save_folder = os.path.join(OCCL_ROOT, "preview")
    os.makedirs(save_folder, exist_ok=True)

    img_paths = sorted(glob.glob(os.path.join(original_folder, "*")))
    for i, orig_path in enumerate(img_paths[:n]):
        fname = os.path.basename(orig_path)
        stem, _ = os.path.splitext(fname)
        lbl_path = os.path.join(label_folder, stem + ".txt")

        orig = cv2.imread(orig_path)
        if orig is None:
            continue

        labels = read_yolo_labels(lbl_path)
        after = add_realistic_submersion(orig, labels)

        combined = np.hstack([orig, after])
        cv2.imwrite(os.path.join(save_folder, f"compare_{i+1}.jpg"), combined)


def train_occlusion_model():
    """Trains YOLO11n on the occlusion-only dataset."""
    data_yaml = os.path.abspath(os.path.join(OCCL_ROOT, "data.yaml"))
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
    build_occlusion_dataset()
    preview_augmented_samples(n=5)

    resp = input("Proceed to training? [y/N]: ").strip().lower()
    if resp == "y":
        train_occlusion_model()
    else:
        print("Skipped training.")


if __name__ == "__main__":
    main()
