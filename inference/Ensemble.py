import os
import sys
import glob

import cv2
import numpy as np
from ultralytics import YOLO


# Phase 2 model weights only
MODEL_PATHS = [
    "/deac/sta/classes/sta379/adena24/runs_phase2_ablation/yolo11n_turbidity_only/weights/best.pt",
    "/deac/sta/classes/sta379/adena24/runs_phase2_ablation/yolo11n_color_only/weights/best.pt",
    "/deac/sta/classes/sta379/adena24/runs_phase2_ablation/yolo11n_glare_only/weights/best.pt",
    "/deac/sta/classes/sta379/adena24/runs_phase2_ablation/yolo11n_occlusion_only_realistic/weights/best.pt",
]

# 0 = drowning, 1 = normal (from your data.yaml)
CLASS_NAMES = {0: "drowning", 1: "normal"}

CONF_THRES = 0.20
IOU_THRES = 0.50


def load_models(paths):
    """Load models that exist; skip missing weights."""
    models = []
    used = []
    for p in paths:
        if os.path.exists(p):
            models.append(YOLO(p))
            used.append(p)
    if not models:
        raise RuntimeError("No model weights found. Check MODEL_PATHS.")
    return models, used


def list_images(path):
    """Return list of images from a file or folder."""
    if os.path.isdir(path):
        imgs = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            imgs.extend(glob.glob(os.path.join(path, ext)))
        return sorted(imgs)
    return [path]


def iou_xyxy(a, b):
    """IoU between two boxes in (x1,y1,x2,y2)."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def fuse_boxes(all_boxes, iou_thresh=0.5):
    """
    Simple box fusion per class:
    - group boxes with IoU > threshold
    - average their coordinates and confidence
    all_boxes: list of [x1,y1,x2,y2,score,cls]
    """
    if not all_boxes:
        return []

    boxes = np.array(all_boxes, dtype=float)
    fused = []

    classes = np.unique(boxes[:, 5]).astype(int)

    for c in classes:
        bc = boxes[boxes[:, 5] == c]
        if len(bc) == 0:
            continue

        # sort by confidence descending
        order = np.argsort(bc[:, 4])[::-1]
        bc = bc[order]

        used = np.zeros(len(bc), dtype=bool)

        for i in range(len(bc)):
            if used[i]:
                continue

            base = bc[i]
            group = [base]
            used[i] = True

            for j in range(i + 1, len(bc)):
                if used[j]:
                    continue
                if iou_xyxy(base[:4], bc[j][:4]) > iou_thresh:
                    group.append(bc[j])
                    used[j] = True

            group = np.array(group, dtype=float)
            x1m, y1m, x2m, y2m = group[:, 0].mean(), group[:, 1].mean(), group[:, 2].mean(), group[:, 3].mean()
            sm = group[:, 4].mean()

            fused.append([x1m, y1m, x2m, y2m, sm, float(c)])

    return fused


def predict_all_models(models, img, conf_thres):
    """Run every model on the image and return a flat list of boxes."""
    H, W = img.shape[:2]
    all_boxes = []

    for model in models:
        res = model.predict(img, conf=conf_thres, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            continue

        for b in res.boxes:
            xyxy = b.xyxy[0].cpu().numpy()
            score = float(b.conf[0])
            cls_id = int(b.cls[0])

            x1, y1, x2, y2 = map(float, xyxy)
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(W), x2), min(float(H), y2)

            all_boxes.append([x1, y1, x2, y2, score, cls_id])

    return all_boxes


def draw_fused(img, fused_boxes):
    """Draw fused boxes on the image."""
    out = img.copy()

    for x1, y1, x2, y2, score, cls_id in fused_boxes:
        cls_id = int(cls_id)
        label = CLASS_NAMES.get(cls_id, str(cls_id))

        # drowning red, normal green (BGR)
        color = (0, 0, 255) if cls_id == 0 else (0, 255, 0)

        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(
            out,
            f"{label}:{score:.2f}",
            (int(x1), max(int(y1) - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return out


def process_single_image(models, image_path, output_path):
    """Run ensemble on one image and save the annotated output."""
    if not os.path.exists(image_path):
        print(f"Missing image: {image_path}")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read: {image_path}")
        return

    all_boxes = predict_all_models(models, img, CONF_THRES)
    fused = fuse_boxes(all_boxes, iou_thresh=IOU_THRES)

    out = draw_fused(img, fused)
    cv2.imwrite(output_path, out)
    print(f"Saved: {output_path}")


def main():
    # Usage: python Ensemble.py /path/to/image_or_folder [output_dir]
    if len(sys.argv) < 2:
        print("Usage: python Ensemble.py /path/to/image_or_folder [output_dir]")
        return

    inp = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) >= 3 else "ensemble_outputs"
    os.makedirs(out_dir, exist_ok=True)

    models, used_paths = load_models(MODEL_PATHS)

    # quick sanity print (small + useful)
    print(f"Loaded {len(models)} model(s).")
    for p in used_paths:
        print(f"  - {p}")

    images = list_images(inp)
    if not images:
        print("No images found.")
        return

    for image_path in images:
        fname = os.path.basename(image_path)
        save_path = os.path.join(out_dir, fname)
        process_single_image(models, image_path, save_path)


if __name__ == "__main__":
    main()
