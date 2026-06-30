"""Pre-label downloaded mouse images for YOLO training using SAM.

Pipeline
--------
1. Read images from --img-dir (default: data/images).
2. Resize every image to 640x640.
3. Convert a random 1/5 of images to grayscale, then back to 3-channel BGR
   (matches the actual IR camera output used at inference time).
4. Run YOLOv8n-COCO to detect 'mouse' (COCO class 64).
   - If found  → use the best bounding box as a SAM prompt for a precise mask.
   - If not found → run SAM in auto-mode and pick the largest mask (mouse is
     usually the main subject of downloaded images).
5. Convert the SAM segmentation mask to a YOLO bbox label (class 0) for
   training yolov8n (detection). Pass --format seg for a segmentation polygon
   instead (needed only if you switch to yolov8n-seg).
6. Write processed images  → --out-dir/images/
   Write YOLO label .txt   → --out-dir/labels/
   Write dataset.yaml      → --out-dir/dataset.yaml
7. Save side-by-side preview images to --out-dir/previews/ (use --preview).

Usage
-----
    python tools/label_images_sam.py
    python tools/label_images_sam.py --img-dir data/images --out-dir data/labeled
    python tools/label_images_sam.py --format bbox --preview

Requirements
------------
    pip install ultralytics opencv-python
    (SAM weights sam_b.pt are downloaded automatically on first run)
"""

import argparse
import random
import textwrap
from pathlib import Path

import cv2
import numpy as np
from ultralytics import SAM, YOLO

# COCO class index for 'mouse' in the standard 80-class YOLOv8n model.
MOUSE_CLASS_COCO = 64
TARGET_SIZE = 640
YOLO_CLASS_ID = 0   # single class in our custom dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def mask_to_polygon(mask: np.ndarray) -> list[float] | None:
    """Convert a binary HxW mask to a normalized YOLO segmentation polygon."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 200:
        return None
    epsilon = 0.004 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) < 3:
        return None
    h, w = mask.shape[:2]
    pts: list[float] = []
    for pt in approx.reshape(-1, 2):
        pts.extend([pt[0] / w, pt[1] / h])
    return pts


def mask_to_bbox(mask: np.ndarray) -> list[float] | None:
    """Convert a binary mask to a normalized YOLO bbox [cx, cy, w, h]."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    h, w = mask.shape[:2]
    cx = (x1 + x2) / 2 / w
    cy = (y1 + y2) / 2 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return [cx, cy, bw, bh]


# ---------------------------------------------------------------------------
# Detection + segmentation
# ---------------------------------------------------------------------------

def detect_mouse_box(yolo: YOLO, img_bgr: np.ndarray) -> list[float] | None:
    """Return [x1, y1, x2, y2] of the best mouse detection, or None."""
    results = yolo(img_bgr, verbose=False)
    best_box: list[float] | None = None
    best_conf = 0.0
    for r in results:
        for box in r.boxes:
            if int(box.cls) == MOUSE_CLASS_COCO and float(box.conf) > best_conf:
                best_conf = float(box.conf)
                best_box = box.xyxy[0].tolist()
    return best_box


def get_best_mask(
    sam: SAM,
    img_bgr: np.ndarray,
    box: list[float] | None,
) -> np.ndarray | None:
    """Run SAM with a box prompt (or auto-mode) and return the best binary mask."""
    if box:
        results = sam(img_bgr, bboxes=[box], verbose=False)
    else:
        results = sam(img_bgr, verbose=False)

    if not results or results[0].masks is None:
        return None

    data = results[0].masks.data
    masks = np.asarray(data.cpu() if hasattr(data, "cpu") else data)  # type: ignore[union-attr]

    if box:
        raw = masks[0]
    else:
        # Pick the largest mask — the mouse is usually the dominant subject.
        raw = max(masks, key=lambda m: m.sum())

    # Resize to 640x640 if SAM returned a different resolution.
    if raw.shape != (TARGET_SIZE, TARGET_SIZE):
        raw = cv2.resize(
            raw.astype(np.uint8), (TARGET_SIZE, TARGET_SIZE),
            interpolation=cv2.INTER_NEAREST,
        )
    return (raw > 0.5).astype(np.uint8)


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------

def make_preview(img_bgr: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Return image with green mask overlay; original on left, overlay on right."""
    if mask is None:
        return np.hstack([img_bgr, img_bgr])
    overlay = img_bgr.copy()
    overlay[mask > 0] = (overlay[mask > 0] * 0.5 + np.array([0, 180, 0]) * 0.5).astype(np.uint8)
    return np.hstack([img_bgr, overlay])


# ---------------------------------------------------------------------------
# Interactive review
# ---------------------------------------------------------------------------

class _BboxEditor:
    """Stateful mouse callback for drawing / replacing a bounding box."""

    WIN = "SAM Review"

    def __init__(self) -> None:
        self._drawing = False
        self._p1: tuple[int, int] | None = None
        self._p2: tuple[int, int] | None = None
        self._user_box: list[int] | None = None  # pixel [x1, y1, x2, y2]

    def reset(self) -> None:
        self._drawing = False
        self._p1 = self._p2 = None
        self._user_box = None

    def callback(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drawing = True
            self._p1 = (x, y)
            self._p2 = (x, y)
            self._user_box = None
        elif event == cv2.EVENT_MOUSEMOVE and self._drawing:
            self._p2 = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self._drawing:
            self._drawing = False
            x1, x2 = sorted([self._p1[0], x])  # type: ignore[index]
            y1, y2 = sorted([self._p1[1], y])   # type: ignore[index]
            if x2 - x1 > 5 and y2 - y1 > 5:
                self._user_box = [x1, y1, x2, y2]

    def render(
        self,
        img: np.ndarray,
        mask: np.ndarray | None,
        sam_box_px: list[int] | None,
        title: str,
    ) -> np.ndarray:
        canvas = img.copy()
        h, w = canvas.shape[:2]

        # Semi-transparent green mask overlay (hidden once user starts drawing)
        if mask is not None and self._user_box is None and not self._drawing:
            green = np.zeros_like(canvas)
            green[mask > 0] = (0, 200, 0)
            canvas = cv2.addWeighted(canvas, 0.65, green, 0.35, 0)

        # SAM-derived bbox in yellow (suggestion)
        if sam_box_px and self._user_box is None and not self._drawing:
            x1, y1, x2, y2 = sam_box_px
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 255), 2)

        # Confirmed user-drawn box in bright green
        if self._user_box:
            x1, y1, x2, y2 = self._user_box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Live drag rectangle in cyan
        if self._drawing and self._p1 and self._p2:
            cv2.rectangle(canvas, self._p1, self._p2, (255, 220, 0), 2)

        # Instruction bar at the top
        bar = np.zeros((36, w, 3), dtype=np.uint8)
        cv2.putText(
            bar,
            f"{title}   Enter=confirm   S=skip   drag=redraw box",
            (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (210, 210, 210), 1, cv2.LINE_AA,
        )
        return np.vstack([bar, canvas])

    def user_bbox_normalized(self, img_w: int, img_h: int) -> list[float] | None:
        if not self._user_box:
            return None
        x1, y1, x2, y2 = self._user_box
        return [
            (x1 + x2) / 2 / img_w,
            (y1 + y2) / 2 / img_h,
            (x2 - x1) / img_w,
            (y2 - y1) / img_h,
        ]


def review_label(
    editor: _BboxEditor,
    sam: SAM,
    img: np.ndarray,
    mask: np.ndarray | None,
    sam_bbox_norm: list[float] | None,
    label_format: str,
    title: str,
) -> tuple[np.ndarray | None, list[float] | None]:
    """
    Show image + SAM result.  User can confirm or redraw.

    Returns (final_mask, final_normalized_bbox_or_polygon) where one of the
    two may be None.  Returns (None, None) when the user skips.
    """
    editor.reset()
    h, w = img.shape[:2]

    # Pre-compute pixel bbox for display
    sam_box_px: list[int] | None = None
    if sam_bbox_norm:
        cx, cy, bw, bh = sam_bbox_norm
        sam_box_px = [
            int((cx - bw / 2) * w), int((cy - bh / 2) * h),
            int((cx + bw / 2) * w), int((cy + bh / 2) * h),
        ]

    cv2.namedWindow(editor.WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(editor.WIN, 700, 740)
    cv2.setMouseCallback(editor.WIN, editor.callback)

    active_mask = mask
    active_bbox_norm = sam_bbox_norm

    while True:
        frame = editor.render(img, active_mask, sam_box_px, title)
        cv2.imshow(editor.WIN, frame)
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 10):  # Enter — confirm
            # If user drew a new box, update SAM mask too (for seg format).
            user_norm = editor.user_bbox_normalized(w, h)
            if user_norm is not None:
                if label_format == "seg":
                    # Re-run SAM with user-drawn box for a better mask.
                    new_mask = get_best_mask(sam, img, [
                        int((user_norm[0] - user_norm[2] / 2) * w),
                        int((user_norm[1] - user_norm[3] / 2) * h),
                        int((user_norm[0] + user_norm[2] / 2) * w),
                        int((user_norm[1] + user_norm[3] / 2) * h),
                    ])
                    active_mask = new_mask
                    active_bbox_norm = user_norm
                else:
                    active_bbox_norm = user_norm
            break

        if key in (ord('s'), ord('S')):  # Skip
            return None, None

    return active_mask, active_bbox_norm


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process(
    img_dir: Path,
    out_dir: Path,
    label_format: str,
    grayscale_fraction: float,
    preview: bool,
    interactive: bool,
) -> None:
    images_out = out_dir / "images"
    labels_out = out_dir / "labels"
    previews_out = out_dir / "previews"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    if preview:
        previews_out.mkdir(parents=True, exist_ok=True)

    paths = sorted(p for p in img_dir.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        print(f"No images found in {img_dir}")
        return

    n_gray = max(1, round(len(paths) * grayscale_fraction))
    gray_indices = set(random.sample(range(len(paths)), n_gray))
    print(f"Total images: {len(paths)}, grayscale: {n_gray}")

    print("Loading YOLOv8n (COCO) …")
    yolo = YOLO("yolov8n.pt")
    print("Loading SAM (sam_b.pt) …")
    sam = SAM("sam_b.pt")

    editor = _BboxEditor() if interactive else None
    if interactive:
        print("Interactive mode: Enter=confirm  S=skip  drag=redraw box")
        print("Close the window or Ctrl+C to abort.")

    labeled = empty = skipped = 0

    for idx, path in enumerate(paths):
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            skipped += 1
            continue

        img_bgr = cv2.resize(img_bgr, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LINEAR)

        if idx in gray_indices:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            img_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # 1. Detect mouse with YOLO for a SAM box prompt.
        box = detect_mouse_box(yolo, img_bgr)

        # 2. Segment with SAM.
        mask = get_best_mask(sam, img_bgr, box)

        # 3. Derive initial label from SAM mask.
        sam_bbox_norm: list[float] | None = None
        if mask is not None:
            sam_bbox_norm = mask_to_bbox(mask)

        # 4. Interactive review (or use SAM result directly).
        if interactive and editor is not None:
            title = f"{idx + 1}/{len(paths)}  {path.name}"
            mask, active_bbox_norm = review_label(
                editor, sam, img_bgr, mask, sam_bbox_norm, label_format, title
            )
        else:
            active_bbox_norm = sam_bbox_norm

        # 5. Build label string.
        label_line = ""
        if label_format == "seg" and mask is not None:
            pts = mask_to_polygon(mask)
            if pts:
                coords = " ".join(f"{v:.6f}" for v in pts)
                label_line = f"{YOLO_CLASS_ID} {coords}"
        elif label_format == "bbox" and active_bbox_norm is not None:
            coords = " ".join(f"{v:.6f}" for v in active_bbox_norm)
            label_line = f"{YOLO_CLASS_ID} {coords}"

        # 6. Save image.
        stem = f"{path.stem}_{idx}"
        out_img = images_out / f"{stem}.jpg"
        cv2.imwrite(str(out_img), img_bgr)

        # 7. Save label.
        label_path = labels_out / f"{stem}.txt"
        label_path.write_text(label_line + "\n" if label_line else "")

        if label_line:
            labeled += 1
        else:
            empty += 1

        # 8. Optional preview.
        if preview:
            prev = make_preview(img_bgr, mask)
            cv2.imwrite(str(previews_out / f"{stem}.jpg"), prev)

        if (idx + 1) % 20 == 0 or (idx + 1) == len(paths):
            status = "gray" if idx in gray_indices else "color"
            box_tag = "YOLO+SAM" if box else "SAM-auto"
            print(f"  [{idx+1:>4}/{len(paths)}] {path.name} | {status} | {box_tag} | labeled={labeled}")

    if interactive:
        cv2.destroyAllWindows()

    # 9. Write dataset.yaml.
    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text(textwrap.dedent(f"""\
        path: {out_dir.resolve().as_posix()}
        train: images
        val: images

        nc: 1
        names:
          0: mouse
    """))

    print(f"\nDone.")
    print(f"  Labeled  : {labeled}")
    print(f"  Empty    : {empty}  (no mask found — review these)")
    print(f"  Skipped  : {skipped}  (unreadable files)")
    print(f"  Images   → {images_out}")
    print(f"  Labels   → {labels_out}")
    print(f"  YAML     → {yaml_path}")
    if preview:
        print(f"  Previews → {previews_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-label mouse images with SAM for YOLO training.")
    parser.add_argument("--img-dir", default="data/images", help="Folder with downloaded images.")
    parser.add_argument("--out-dir", default="data/labeled", help="Output folder for images, labels, yaml.")
    parser.add_argument(
        "--format",
        choices=["bbox", "seg"],
        default="bbox",
        help="Label format: 'bbox' = bounding box for YOLOv8n detection (default), 'seg' = segmentation polygon.",
    )
    parser.add_argument(
        "--gray-fraction",
        type=float,
        default=0.2,
        help="Fraction of images to convert to grayscale (default: 0.2 = 1/5).",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Skip interactive review and save SAM results directly.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Save side-by-side preview images with mask overlay to out-dir/previews/.",
    )
    args = parser.parse_args()

    process(
        img_dir=Path(args.img_dir),
        out_dir=Path(args.out_dir),
        label_format=args.format,
        grayscale_fraction=args.gray_fraction,
        preview=args.preview,
        interactive=not args.no_interactive,
    )


if __name__ == "__main__":
    main()
