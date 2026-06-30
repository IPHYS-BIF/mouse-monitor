"""
YOLOv8n Video Object Detection Training Pipeline
=================================================

Full workflow for training a YOLOv8n model using video data:

  1. extract_frames()             -> pull frames out of video(s) into images for labeling
  2. split_dataset()               -> organize labeled images/labels into train/val folders
     OR build_dataset_with_negatives() -> for single-class + hard-negative folders (e.g. mouse/room/other_animal)
  3. write_data_yaml()             -> generate the data.yaml config YOLO needs
  4. train_model()                 -> fine-tune yolov8n.pt on your dataset
  5. validate_model()              -> run validation metrics (mAP, precision, recall)
  6. run_inference_on_video()      -> use the trained model to detect objects in a video

Install dependencies first:
    pip install ultralytics opencv-python --break-system-packages

IMPORTANT: Step 1 only extracts frames. You still need to LABEL them
(draw bounding boxes + class) using a tool such as:
    - LabelImg        (https://github.com/heartexlab/labelImg)
    - CVAT            (https://cvat.ai)
    - Roboflow        (https://roboflow.com) - can also auto-export YOLO format
    - Label Studio    (https://labelstud.io)

Each labeled image needs a matching .txt file in YOLO format:
    <class_id> <x_center> <y_center> <width> <height>   (all normalized 0-1)
"""

import os
import shutil
import random
from pathlib import Path

import cv2
from ultralytics import YOLO


# ----------------------------------------------------------------------
# 1. EXTRACT FRAMES FROM VIDEO
# ----------------------------------------------------------------------
def extract_frames(video_path: str, output_dir: str, frame_interval: int = 15,
                    max_frames: int = None, prefix: str = None):
    """
    Extract frames from a video at a fixed interval to use as training images.

    Args:
        video_path: path to the input video file
        output_dir: folder where extracted frames (.jpg) will be saved
        frame_interval: save 1 out of every N frames (e.g. 15 = ~2 frames/sec at 30fps)
        max_frames: optional cap on number of frames to extract
        prefix: filename prefix (defaults to video filename, useful when
                 extracting from multiple videos into the same folder)
    """
    os.makedirs(output_dir, exist_ok=True)
    prefix = prefix or Path(video_path).stem

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[extract_frames] {video_path}: {total_frames} frames @ {fps:.1f} fps")

    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            filename = f"{prefix}_frame_{frame_idx:06d}.jpg"
            out_path = os.path.join(output_dir, filename)
            cv2.imwrite(out_path, frame)
            saved_count += 1

            if max_frames and saved_count >= max_frames:
                break

        frame_idx += 1

    cap.release()
    print(f"[extract_frames] Saved {saved_count} frames to {output_dir}")
    return saved_count


def extract_frames_from_multiple_videos(video_paths: list, output_dir: str,
                                         frame_interval: int = 15, max_frames_per_video: int = None):
    """Convenience wrapper to extract frames from several videos into one folder."""
    total = 0
    for vp in video_paths:
        total += extract_frames(vp, output_dir, frame_interval, max_frames_per_video)
    print(f"[extract_frames_from_multiple_videos] Total frames extracted: {total}")
    return total


# ----------------------------------------------------------------------
# 2. SPLIT INTO TRAIN / VAL (after you've labeled the extracted frames)
# ----------------------------------------------------------------------
def split_dataset(images_dir: str, labels_dir: str, dataset_root: str,
                   val_ratio: float = 0.2, seed: int = 42):
    """
    Organize labeled images + YOLO .txt label files into the structure
    ultralytics expects:

        dataset_root/
            images/train/*.jpg
            images/val/*.jpg
            labels/train/*.txt
            labels/val/*.txt

    Only images that have a matching label file are included
    (unlabeled frames are skipped).
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    dataset_root = Path(dataset_root)

    img_exts = {".jpg", ".jpeg", ".png"}
    all_images = [p for p in images_dir.iterdir() if p.suffix.lower() in img_exts]

    # Keep only images that have a corresponding label file
    paired = []
    for img_path in all_images:
        label_path = labels_dir / (img_path.stem + ".txt")
        if label_path.exists():
            paired.append((img_path, label_path))
        else:
            print(f"[split_dataset] WARNING: no label for {img_path.name}, skipping")

    if not paired:
        raise ValueError("No labeled image/label pairs found. Label your frames first.")

    random.seed(seed)
    random.shuffle(paired)

    val_count = max(1, int(len(paired) * val_ratio))
    val_set = paired[:val_count]
    train_set = paired[val_count:]

    for split_name, split_data in [("train", train_set), ("val", val_set)]:
        img_out = dataset_root / "images" / split_name
        lbl_out = dataset_root / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path, label_path in split_data:
            shutil.copy2(img_path, img_out / img_path.name)
            shutil.copy2(label_path, lbl_out / label_path.name)

    print(f"[split_dataset] Train: {len(train_set)} images | Val: {len(val_set)} images")
    return dataset_root


# ----------------------------------------------------------------------
# 2b. BUILD DATASET FROM POSITIVE + NEGATIVE SOURCES (e.g. mouse / room / other_animal)
# ----------------------------------------------------------------------
def build_dataset_with_negatives(positive_images_dir: str, positive_labels_dir: str,
                                  negative_dirs: list, dataset_root: str,
                                  val_ratio: float = 0.2,
                                  negative_fraction_of_positives: float = 0.2,
                                  seed: int = 42):
    """
    Assemble a YOLO dataset from one labeled positive class (e.g. 'mouse') plus
    one or more negative/background image folders (e.g. 'room', 'other_animal')
    that contain NO instances of the target class.

    Negative images get an EMPTY label file (no boxes) so YOLO learns to
    suppress false positives on background and look-alike objects, without
    needing any annotation work on those folders.

    Args:
        positive_images_dir: folder of labeled images (e.g. 'mouse/images')
        positive_labels_dir: folder of matching YOLO .txt label files
        negative_dirs: list of folders containing only images, no labels
                        (e.g. ['room/images', 'other_animal/images'])
        dataset_root: output dataset folder
        val_ratio: fraction of POSITIVE images held out for validation
                    (val set is kept mostly/fully positive so mAP is meaningful)
        negative_fraction_of_positives: how many negatives to include,
                    expressed as a fraction of the positive image count.
                    e.g. 0.2 with 1000 positive images -> ~200 negatives total,
                    split evenly across negative_dirs
        seed: random seed for reproducible splits
    """
    random.seed(seed)
    dataset_root = Path(dataset_root)
    pos_img_dir = Path(positive_images_dir)
    pos_lbl_dir = Path(positive_labels_dir)

    img_exts = {".jpg", ".jpeg", ".png"}

    # --- gather positive pairs ---
    pos_images = [p for p in pos_img_dir.iterdir() if p.suffix.lower() in img_exts]
    pos_pairs = []
    for img_path in pos_images:
        label_path = pos_lbl_dir / (img_path.stem + ".txt")
        if label_path.exists():
            pos_pairs.append((img_path, label_path))
        else:
            print(f"[build_dataset_with_negatives] WARNING: no label for {img_path.name}, skipping")

    if not pos_pairs:
        raise ValueError("No labeled positive image/label pairs found.")

    random.shuffle(pos_pairs)
    val_count = max(1, int(len(pos_pairs) * val_ratio))
    pos_val = pos_pairs[:val_count]
    pos_train = pos_pairs[val_count:]

    # --- gather negative images, split evenly across provided folders ---
    target_total_negatives = int(len(pos_pairs) * negative_fraction_of_positives)
    per_folder_count = target_total_negatives // max(1, len(negative_dirs))

    neg_images = []
    for neg_dir in negative_dirs:
        neg_dir = Path(neg_dir)
        candidates = [p for p in neg_dir.iterdir() if p.suffix.lower() in img_exts]
        random.shuffle(candidates)
        chosen = candidates[:per_folder_count]
        neg_images.extend(chosen)
        print(f"[build_dataset_with_negatives] {neg_dir}: using {len(chosen)}/{len(candidates)} images as negatives")

    # negatives go (almost) entirely into train; keep val mostly clean/positive
    random.shuffle(neg_images)
    neg_val_count = max(0, int(len(neg_images) * 0.05))  # tiny slice in val to sanity-check false positives
    neg_val = neg_images[:neg_val_count]
    neg_train = neg_images[neg_val_count:]

    # --- write out splits ---
    def write_split(split_name, pos_split, neg_split):
        img_out = dataset_root / "images" / split_name
        lbl_out = dataset_root / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path, label_path in pos_split:
            shutil.copy2(img_path, img_out / img_path.name)
            shutil.copy2(label_path, lbl_out / label_path.name)

        for img_path in neg_split:
            shutil.copy2(img_path, img_out / img_path.name)
            (lbl_out / (img_path.stem + ".txt")).write_text("")  # empty = no objects

        print(f"[build_dataset_with_negatives] {split_name}: {len(pos_split)} positive + {len(neg_split)} negative images")

    write_split("train", pos_train, neg_train)
    write_split("val", pos_val, neg_val)

    print(f"[build_dataset_with_negatives] Done. Dataset at {dataset_root}")
    return dataset_root


# ----------------------------------------------------------------------
# 3. WRITE data.yaml
# ----------------------------------------------------------------------
def write_data_yaml(dataset_root: str, class_names: list, yaml_path: str = None):
    """
    Write the data.yaml config file YOLO needs for training.

    Args:
        dataset_root: root folder containing images/ and labels/ subfolders
        class_names: ordered list of class names, e.g. ["person", "car", "dog"]
                     index in this list = class_id used in your .txt label files
        yaml_path: where to save the yaml (defaults to dataset_root/data.yaml)
    """
    dataset_root = Path(dataset_root).resolve()
    yaml_path = Path(yaml_path) if yaml_path else dataset_root / "data.yaml"

    content = (
        f"path: {dataset_root}\n"
        f"train: images/train\n"
        f"val: images/val\n\n"
        f"nc: {len(class_names)}\n"
        f"names: {class_names}\n"
    )

    yaml_path.write_text(content)
    print(f"[write_data_yaml] Wrote {yaml_path}")
    return str(yaml_path)


# ----------------------------------------------------------------------
# 4. TRAIN
# ----------------------------------------------------------------------
def train_model(data_yaml: str, epochs: int = 150, imgsz: int = 640,
                 batch: int = -1, model_weights: str = "yolov8n.pt",
                 project: str = "runs/detect", name: str = "train", device=0,
                 patience: int = 20,
                 mosaic: float = 1.0, mixup: float = 0.1,
                 degrees: float = 10.0, flipud: float = 0.5, fliplr: float = 0.5,
                 hsv_h: float = 0.015, hsv_s: float = 0.5, hsv_v: float = 0.3,
                 **extra_train_kwargs):
    """
    Fine-tune YOLOv8n on your custom video-derived dataset.

    Defaults below are tuned for a SMALL single-class dataset (e.g. ~600
    positive + ~120 negative images, as in a mouse-detection setup) where
    augmentation needs to do more of the generalization work and overfitting
    is a real risk.

    Args:
        data_yaml: path to data.yaml
        epochs: number of training epochs (early stopping via `patience` will
                 usually cut this short for small datasets)
        imgsz: training image size (640 is the right default for yolov8n)
        batch: batch size. -1 = auto-select largest batch that fits ~60% of
                available GPU VRAM (recommended, e.g. on RTX 5080 16GB)
        model_weights: starting weights ('yolov8n.pt' = pretrained COCO weights,
                        recommended for transfer learning instead of training from scratch)
        device: e.g. 0 for first GPU, 'cpu' for CPU
        patience: epochs with no val improvement before early stopping

        --- augmentation (tuned for small datasets / mouse-like targets) ---
        mosaic: mosaic augmentation probability (combines 4 images into one,
                 strong regularizer, keep at 1.0 for small datasets)
        mixup: blends two images+labels together; small datasets benefit from
                a light amount (0.1) to discourage memorization
        degrees: max rotation jitter in degrees. Mice can appear at any
                  orientation (unlike e.g. pedestrians/cars), so some rotation
                  augmentation helps generalize across viewing angles
        flipud: vertical flip probability. Off by default in most YOLO presets
                 (assumes upright scenes), but worth enabling for top-down/
                 any-orientation subjects like a mouse in an enclosure
        fliplr: horizontal flip probability (standard, left/right symmetry)
        hsv_h, hsv_s, hsv_v: hue/saturation/value jitter for lighting and
                 color robustness (helps with varying room lighting conditions)
        extra_train_kwargs: any additional ultralytics `model.train()` kwargs
                 you want to pass through (e.g. lr0, optimizer, cos_lr, etc.)
    """
    model = YOLO(model_weights)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        device=device,
        patience=patience,
        save=True,
        plots=True,
        mosaic=mosaic,
        mixup=mixup,
        degrees=degrees,
        flipud=flipud,
        fliplr=fliplr,
        hsv_h=hsv_h,
        hsv_s=hsv_s,
        hsv_v=hsv_v,
        **extra_train_kwargs,
    )

    best_weights = Path(project) / name / "weights" / "best.pt"
    print(f"[train_model] Training complete. Best weights: {best_weights}")
    return str(best_weights)


# ----------------------------------------------------------------------
# 5. VALIDATE
# ----------------------------------------------------------------------
def validate_model(weights_path: str, data_yaml: str):
    """Run validation and print mAP/precision/recall metrics."""
    model = YOLO(weights_path)
    metrics = model.val(data=data_yaml)
    print(f"[validate_model] mAP50-95: {metrics.box.map:.4f} | mAP50: {metrics.box.map50:.4f}")
    return metrics


# ----------------------------------------------------------------------
# 6. RUN INFERENCE ON A NEW VIDEO
# ----------------------------------------------------------------------
def run_inference_on_video(weights_path: str, video_path: str, output_dir: str = "runs/detect/predict",
                            conf: float = 0.25, save_video: bool = True):
    """
    Run the trained model on a video and save annotated output.
    """
    model = YOLO(weights_path)
    results = model.predict(
        source=video_path,
        conf=conf,
        save=save_video,
        project=Path(output_dir).parent.as_posix(),
        name=Path(output_dir).name,
    )
    print(f"[run_inference_on_video] Done. Output saved under {output_dir}")
    return results


# ----------------------------------------------------------------------
# EXAMPLE END-TO-END USAGE
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # --- Step 1: extract frames from your raw videos ---
    extract_frames_from_multiple_videos(
        video_paths=["videos/clip1.mp4", "videos/clip2.mp4"],
        output_dir="dataset_raw/images",
        frame_interval=15,        # adjust based on video length/fps
        max_frames_per_video=500,
    )

    # --- STOP HERE: label "dataset_raw/images" frames now using LabelImg/
    #     CVAT/Roboflow/Label Studio, exporting YOLO-format .txt files
    #     into "dataset_raw/labels" (same filename stem as each image). ---

    # --- Step 2 (single-class with hard negatives, e.g. mouse detection): ---
    # Use this instead of split_dataset() when you have a labeled positive
    # set plus unlabeled negative/background folders (room, other_animal, etc.)
    # Sized here for ~600 positive (mouse) + ~120 negative (room + other_animal)
    # images, i.e. negative_fraction_of_positives=0.2 -> ~120 negatives total.
    dataset_root = build_dataset_with_negatives(
        positive_images_dir="mouse/images",
        positive_labels_dir="mouse/labels",
        negative_dirs=["room/images", "other_animal/images"],
        dataset_root="dataset",
        val_ratio=0.2,
        negative_fraction_of_positives=0.2,  # ~120 negatives for ~600 positives
    )

    # --- Step 3: write data.yaml ---
    data_yaml = write_data_yaml(
        dataset_root=dataset_root,
        class_names=["mouse"],  # single class
    )

    # --- Step 4: train ---
    # epochs/batch/augmentation defaults in train_model() are already tuned
    # for this dataset size (small, single-class, any-orientation target).
    best_weights = train_model(
        data_yaml=data_yaml,
        model_weights="yolov8n.pt",
        device=0,           # RTX 5080
        # batch=-1 (auto), epochs=150, patience=20, mosaic/mixup/flipud/etc.
        # all use the tuned defaults above — override here if needed, e.g.:
        # lr0=0.005, cos_lr=True,
    )

    # --- Step 5: validate ---
    validate_model(best_weights, data_yaml)

    # --- Step 6: run on a new video ---
    run_inference_on_video(best_weights, "videos/test_clip.mp4")