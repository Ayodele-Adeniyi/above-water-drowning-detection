import os
import glob
import random
from shutil import copy2

import numpy as np
import cv2
import yaml
from ultralytics import YOLO


# Original dataset (Roboflow export)
ORIG_ROOT = "/deac/sta/classes/sta379/adena24/AboveWater_Drowning_Detection.v1i.yolov11"
ORIG_DATA_YAML = os.path.join(ORIG_ROOT, "data.yaml")

# New dataset root (Phase 3 combined augs)
COMB_ROOT = "/deac/sta/classes/sta379/adena24/AboveWater_Drowning_Detection_Combined"

SPLITS = ["train", "valid", "test"]

PROJECT_NAME = "runs_phase3_robust"
RUN_NAME = "yolo11n_phase3_combined"

# Save a few side-by-side previews (original | augmented)
SAVE_PREVIEW = True
PREVIEW_DIR = os.path.join(COMB_ROOT, "_previews")


def add_water_turbidity_haze(img, strength_range=(0.15, 0.4), particle_intensity=0.02, flip_vertical_prob=0.3):
    """Adds haze + mild particulate noise to mimic turbid water."""
    img_f = img.astype(np.float32) / 255.0

    alpha = random.uniform(*strength_range)

    base = np.array([1.0, 0.95, 0.9], dtype=np.float32)  # slightly warm haze
    jitter = np.random.uniform(-0.05, 0.05, size=(3,))
    haze_color = np.clip(base + jitter, 0.8, 1.0)

    h, w, _ = img_f.shape
    y = np.linspace(0, 1, h).reshape(-1, 1, 1)

    alpha_map = alpha * y
    if random.random() < flip_vertical_prob:
        alpha_map = alpha * (1.0 - y)

    alpha_map = np.repeat(alpha_map, 3, axis=2)
    hazy = img_f * (1 - alpha_map) + haze_color * alpha_map

    if particle_intensity > 0:
        hazy += np.random.normal(0.0, particle_intensity, size=img_f.shape)

    hazy = np.clip(hazy, 0, 1)
    hazy = cv2.GaussianBlur(hazy, (5, 5), 0)
    return (hazy * 255).astype(np.uint8)


def add_underwater_color_distortion(
    img,
    red_atten_range=(0.3, 0.7),
    blue_boost_range=(0.0, 0.2),
    green_boost_range=(0.0, 0.15),
):
    """Attenuates red and boosts blue/green with depth (vertical gradient)."""
    img_f = img.astype(np.float32) / 255.0
    h, w, _ = img_f.shape

    y = np.linspace(0, 1, h).reshape(-1, 1)

    red_att = random.uniform(*red_atten_range)
    blue_boost = random.uniform(*blue_boost_range)
    green_boost = random.uniform(*green_boost_range)

    B, G, R = img_f[:, :, 0], img_f[:, :, 1], img_f[:, :, 2]

    red_scale = 1.0 - red_att * y
    red_scale = np.repeat(red_scale, w, axis=1)
    R = R * red_scale

    blue_scale = 1.0 + blue_boost * y
    green_scale = 1.0 + green_boost * y
    blue_scale = np.repeat(blue_scale, w, axis=1)
    green_scale = np.repeat(green_scale, w, axis=1)

    B = np.clip(B * blue_scale, 0.0, 1.0)
    G = np.clip(G * green_scale, 0.0, 1.0)

    out = np.stack([B, G, R], axis=2)
    out = cv2.GaussianBlur(out, (3, 3), 0)
    out = np.clip(out, 0.0, 1.0)
    return (out * 255).astype(np.uint8)


def add_surface_glare(img, max_patches=2, intensity_range=(0.55, 0.85), size_ratio_range=(0.15, 0.3)):
    """Adds bright specular patches near the top area (surface glare)."""
    h, w, _ = img.shape
    img_f = img.astype(np.float32) / 255.0

    glare_mask = np.zeros((h, w), dtype=np.float32)
    n_patches = random.randint(1, max_patches)

    x = np.arange(w)
    y = np.arange(h)
    xx, yy = np.meshgrid(x, y)

    for _ in range(n_patches):
        patch_w = int(w * random.uniform(*size_ratio_range))
        patch_h = int(h * random.uniform(0.05, 0.12))

        cx = random.randint(patch_w // 2, w - patch_w // 2)
        cy = random.randint(int(h * 0.02), int(h * 0.35))

        sx = patch_w / 2.0
        sy = patch_h / 2.0

        gauss = np.exp(-(((xx - cx) ** 2) / (2 * sx ** 2) + ((yy - cy) ** 2) / (2 * sy ** 2)))
        gauss /= gauss.max() + 1e-8

        strength = random.uniform(*intensity_range)
        glare_mask = np.clip(glare_mask + strength * gauss, 0.0, 1.0)

    glare_mask_3 = np.stack([glare_mask] * 3, axis=2)

    # Blend toward white where glare mask is high
    out = img_f * (1 - glare_mask_3 * 0.7) + 1.0 * (glare_mask_3 * 0.7)
    out = np.clip(out, 0.0, 1.0)
    return (out * 255).astype(np.uint8)


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
    Tries to "partially submerge" part of the person using the first label:
    - blur + desaturate + slight tint
    - ripple warp
    - soft feathered mask, blended into the original
    """
    if not labels:
        return img

    img_f = img.astype(np.float32) / 255.0
    H, W, _ = img_f.shape

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

    y0 = px_cy  # approx waist
    y1 = min(H, px_cy + sub_pixels)
    if y1 <= y0:
        return img

    x0 = max(0, px_cx - int(px_w * 0.6))
    x1 = min(W, px_cx + int(px_w * 0.6))

    region = img_f[y0:y1, x0:x1]
    Rh, Rw, _ = region.shape
    if Rh <= 0 or Rw <= 0:
        return img

    # Build a soft mask (ellipse + blur + small noise)
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
    noise = cv2.GaussianBlur(np.random.uniform(0, 0.25, (Rh, Rw)).astype(np.float32), (kf, kf), 0)
    mask = np.clip(mask + noise * 0.2, 0, 1)

    # Blur a bit to mimic underwater loss of detail
    k = random.randint(*blur_strength_range)
    if k % 2 == 0:
        k += 1
    region_blur = cv2.GaussianBlur(region, (k, k), 0)

    # Slight desaturation
    gray = cv2.cvtColor((region_blur * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY) / 255.0
    gray3 = np.stack([gray, gray, gray], axis=2)
    region_desat = region_blur * (1 - desaturate_strength) + gray3 * desaturate_strength

    # Slight green/blue tint
    tint = np.array([0.0, tint_strength, tint_strength]).reshape(1, 1, 3)
    region_tinted = np.clip(region_desat + tint, 0, 1)

    # Small ripple warp
    ry_grid, rx_grid = np.indices((Rh, Rw))
    ripple = (np.sin(rx_grid / 15) + np.cos(ry_grid / 22)) * ripple_strength
    yy2 = np.clip(ry_grid + ripple.astype(int), 0, Rh - 1)
    xx2 = np.clip(rx_grid + ripple.astype(int), 0, Rw - 1)
    region_warped = region_tinted[yy2, xx2]

    # Blend warped region back into the image using the soft mask
    mask3 = np.repeat(mask[:, :, None], 3, axis=2)
    out = img_f.copy()
    out[y0:y1, x0:x1] = region * (1 - mask3) + region_warped * mask3

    out = np.clip(out, 0.0, 1.0)
    return (out * 255).astype(np.uint8)


def read_yolo_labels(label_path):
    """Reads YOLO txt labels as floats: [cls, cx, cy, w, h]."""
    if (not os.path.exists(label_path)) or os.path.getsize(label_path) == 0:
        return []
    labels = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            labels.append(list(map(float, parts)))
    return labels


def apply_random_augmentations(img, labels):
    """Randomly applies a subset of the four augmentations."""
    if random.random() < 0.7:
        img = add_water_turbidity_haze(img)
    if random.random() < 0.6:
        img = add_underwater_color_distortion(img)
    if random.random() < 0.5:
        img = add_surface_glare(img)
    if random.random() < 0.5:
        img = add_realistic_submersion(img, labels)
    return img


def build_combined_dataset():
    """Builds COMB_ROOT dataset: augmented train, unchanged valid/test, labels copied."""
    for split in SPLITS:
        for sub in ["images", "labels"]:
            os.makedirs(os.path.join(COMB_ROOT, split, sub), exist_ok=True)

    if SAVE_PREVIEW:
        os.makedirs(PREVIEW_DIR, exist_ok=True)

    preview_count = 0

    for split in SPLITS:
        src_img_dir = os.path.join(ORIG_ROOT, split, "images")
        src_lbl_dir = os.path.join(ORIG_ROOT, split, "labels")
        dst_img_dir = os.path.join(COMB_ROOT, split, "images")
        dst_lbl_dir = os.path.join(COMB_ROOT, split, "labels")

        img_paths = sorted(glob.glob(os.path.join(src_img_dir, "*.*")))

        for img_path in img_paths:
            fname = os.path.basename(img_path)
            stem, _ = os.path.splitext(fname)

            src_lbl_path = os.path.join(src_lbl_dir, stem + ".txt")
            dst_lbl_path = os.path.join(dst_lbl_dir, stem + ".txt")
            dst_img_path = os.path.join(dst_img_dir, fname)

            if os.path.exists(src_lbl_path):
                copy2(src_lbl_path, dst_lbl_path)

            # Keep valid/test untouched (train only gets combined augmentation)
            if split in ("valid", "test"):
                copy2(img_path, dst_img_path)
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue

            labels = read_yolo_labels(src_lbl_path)
            aug = apply_random_augmentations(img, labels)
            cv2.imwrite(dst_img_path, aug)

            if SAVE_PREVIEW and preview_count < 6:
                comp = np.hstack([img, aug])
                out_prev = os.path.join(PREVIEW_DIR, f"preview_{split}_{preview_count}.jpg")
                cv2.imwrite(out_prev, comp)
                preview_count += 1

    # Write a fresh data.yaml that points at the combined dataset
    new_yaml_path = os.path.join(COMB_ROOT, "data.yaml")
    with open(ORIG_DATA_YAML, "r") as f:
        data = yaml.safe_load(f)

    data["path"] = os.path.abspath(COMB_ROOT)
    data["train"] = "train/images"
    data["val"] = "valid/images"
    data["test"] = "test/images"

    with open(new_yaml_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def train_combined_model():
    """Trains YOLO11n on the Phase 3 combined dataset."""
    data_yaml = os.path.abspath(os.path.join(COMB_ROOT, "data.yaml"))
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
    build_combined_dataset()

    if SAVE_PREVIEW:
        print(f"Previews saved in: {PREVIEW_DIR}")

    train_combined_model()


if __name__ == "__main__":
    main()
