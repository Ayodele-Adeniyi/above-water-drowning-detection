import os
import sys
import glob

import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO


# Update these to your trained weights
MODEL_PATHS = [
    "/deac/sta/classes/sta379/adena24/runs_phase2_ablation/yolo11n_turbidity_only/weights/best.pt",
    "/deac/sta/classes/sta379/adena24/runs_phase2_ablation/yolo11n_color_only/weights/best.pt",
    "/deac/sta/classes/sta379/adena24/runs_phase2_ablation/yolo11n_glare_only/weights/best.pt",
    "/deac/sta/classes/sta379/adena24/runs_phase2_ablation/yolo11n_occlusion_only_realistic/weights/best.pt",
    "/deac/sta/classes/sta379/adena24/runs_phase3_robust/yolo11n_phase3_combined/weights/best.pt",
]

# Confidence threshold used when collecting detections from each model
CONF_THRES = 0.25

# From your data.yaml: 0 = drowning, 1 = normal
CLASS_NAMES = {0: "DROWNING", 1: "NORMAL"}


def load_models(paths):
    """Loads YOLO models that exist on disk; missing weights are skipped."""
    models = []
    used_paths = []
    for p in paths:
        if os.path.exists(p):
            models.append(YOLO(p))
            used_paths.append(p)
    if not models:
        raise RuntimeError("No model weights found. Check MODEL_PATHS.")
    return models, used_paths


def list_images(inp_path):
    """Returns a list of image paths (single file or all images in a folder)."""
    if os.path.isdir(inp_path):
        imgs = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            imgs.extend(glob.glob(os.path.join(inp_path, ext)))
        return sorted(imgs)
    return [inp_path]


def show_image(title, bgr_img):
    """Displays an image using matplotlib (expects BGR input from OpenCV)."""
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 6))
    plt.imshow(rgb)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def draw_boxes(bgr_img, detections):
    """Draws bounding boxes from a list of detections onto a copy of the image."""
    out = bgr_img.copy()

    for det in detections:
        x1, y1, x2, y2 = map(int, [det["x1"], det["y1"], det["x2"], det["y2"]])
        conf = det["conf"]
        cls = int(det["cls"])
        label = CLASS_NAMES.get(cls, str(cls))

        # Color: drowning red, normal green (OpenCV uses BGR)
        color = (0, 0, 255) if label == "DROWNING" else (0, 255, 0)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out,
            f"{label} {conf:.2f}",
            (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return out


def ensemble_vote(detections_by_model):
    """
    One vote per model:
    - each model votes the class of its highest-confidence detection
    - if a model detects nothing, it abstains
    Decision:
    - majority wins
    - tie broken by average confidence
    """
    votes = []
    confs = {0: [], 1: []}

    for dets in detections_by_model:
        if not dets:
            continue
        best = max(dets, key=lambda d: d["conf"])
        cls = int(best["cls"])
        votes.append(cls)
        confs[cls].append(best["conf"])

    if not votes:
        return "NO DETECTION", {"votes": votes, "confs": confs}

    count0 = votes.count(0)
    count1 = votes.count(1)

    if count0 > count1:
        return "DROWNING", {"votes": votes, "confs": confs}
    if count1 > count0:
        return "NORMAL", {"votes": votes, "confs": confs}

    avg0 = float(np.mean(confs[0])) if confs[0] else 0.0
    avg1 = float(np.mean(confs[1])) if confs[1] else 0.0
    final = "DROWNING" if avg0 >= avg1 else "NORMAL"
    return final, {"votes": votes, "confs": confs, "avg0": avg0, "avg1": avg1}


def predict_one_model(model, img, model_idx):
    """Runs prediction and converts YOLO output into a simple list of dicts."""
    res = model.predict(img, conf=CONF_THRES, verbose=False)[0]

    dets = []
    if res.boxes is None or len(res.boxes) == 0:
        return dets

    for b in res.boxes:
        xyxy = b.xyxy[0].cpu().numpy()
        dets.append(
            {
                "x1": float(xyxy[0]),
                "y1": float(xyxy[1]),
                "x2": float(xyxy[2]),
                "y2": float(xyxy[3]),
                "conf": float(b.conf[0]),
                "cls": int(b.cls[0]),
                "model_idx": model_idx,
            }
        )

    return dets


def run_ensemble_on_image(models, image_path, out_dir=None, show_raw=True):
    """Shows raw image, runs ensemble, shows annotated result, optionally saves output."""
    if not os.path.exists(image_path):
        print(f"Missing image: {image_path}")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read: {image_path}")
        return

    if show_raw:
        show_image(f"INPUT: {os.path.basename(image_path)}", img)

    detections_by_model = []
    all_detections = []

    for mi, model in enumerate(models):
        model_dets = predict_one_model(model, img, mi)
        detections_by_model.append(model_dets)
        all_detections.extend(model_dets)

    final_label, debug = ensemble_vote(detections_by_model)

    annotated = draw_boxes(img, all_detections)
    show_image(f"ENSEMBLE: {final_label}", annotated)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, os.path.basename(image_path))
        cv2.imwrite(out_path, annotated)
        print(f"Saved: {out_path}")

    votes = debug.get("votes", [])
    print(f"Image: {image_path}")
    print(f"Final: {final_label}")
    print(f"Votes (0=drowning,1=normal): {votes} | drowning={votes.count(0)} normal={votes.count(1)}")
    if "avg0" in debug:
        print(f"Tie-break avg conf: drowning={debug['avg0']:.3f} normal={debug['avg1']:.3f}")
    print()


def parse_args(argv):
    """Tiny argument parser: input path + optional --out output_dir."""
    if len(argv) < 2:
        print("Usage: python live_demo_ensemble.py /path/to/image_or_folder [--out output_dir]")
        sys.exit(1)

    inp = argv[1]
    out_dir = None

    if "--out" in argv:
        i = argv.index("--out")
        if i + 1 < len(argv):
            out_dir = argv[i + 1]

    return inp, out_dir


def main():
    inp, out_dir = parse_args(sys.argv)

    models, used_paths = load_models(MODEL_PATHS)
    images = list_images(inp)

    print(f"Loaded {len(models)} model(s).")
    for p in used_paths:
        print(f"  - {p}")
    print(f"Found {len(images)} image(s).\n")

    for idx, p in enumerate(images, start=1):
        print(f"[{idx}/{len(images)}] {p}")
        run_ensemble_on_image(models, p, out_dir=out_dir, show_raw=True)

        if idx < len(images):
            input("Enter for next image (Ctrl+C to stop)...")

    print("Done.")


if __name__ == "__main__":
    main()
