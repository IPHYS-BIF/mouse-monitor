"""Pre-label downloaded mouse images for YOLO training using SAM.

Pipeline
--------
Positive dataset (--dataset positive, default):
  1. Read images from --img-dir.
  2. Resize every image to 640x640.
  3. Convert a random 1/5 of images to grayscale (matches IR camera output).
  4. Run YOLOv8n-COCO to detect 'mouse' (COCO class 64) → SAM box prompt.
     Fallback: SAM auto-mode if no mouse detected.
  5. Write YOLO bbox label  <class_id> <cx> <cy> <w> <h>  (normalized, class 0)
     or segmentation polygon with --format seg.
  6. Interactive review: confirm SAM result or drag a new box, S=skip.

Negative dataset (--dataset negative):
  1. Read images from --img-dir (should be background/room images, no mouse).
  2. Resize to 640x640, apply grayscale to 1/5.
  3. Write an EMPTY .txt label file — YOLO treats these as background samples.
  4. Interactive review: confirm each image is truly background, S=skip.

Outputs → --out-dir/images/, --out-dir/labels/, --out-dir/dataset.yaml

Usage
-----
    python tools/label_images_sam.py --dataset positive
    python tools/label_images_sam.py --dataset negative --img-dir data/background
    python tools/label_images_sam.py --dataset positive --format bbox --preview
    python tools/label_images_sam.py --no-interactive

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
import torch
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

def detect_mouse_boxes(yolo: YOLO, img_bgr: np.ndarray, conf: float = 0.25) -> list[list[float]]:
    """Return all [x1, y1, x2, y2] mouse detections above confidence threshold."""
    results = yolo(img_bgr, verbose=False)
    boxes: list[list[float]] = []
    for r in results:
        for box in r.boxes:
            if int(box.cls) == MOUSE_CLASS_COCO and float(box.conf) >= conf:
                boxes.append(box.xyxy[0].tolist())
    return boxes


def get_masks_for_boxes(
    sam: SAM,
    img_bgr: np.ndarray,
    boxes: list[list[float]],
) -> list[np.ndarray]:
    """
    Run SAM with all box prompts at once; return one binary mask per box.
    If no boxes, run in auto-mode and return the single largest mask.
    """
    if boxes:
        results = sam(img_bgr, bboxes=boxes, verbose=False, device="gpu")
    else:
        results = sam(img_bgr, verbose=False, device="gpu")

    if not results or results[0].masks is None:
        return []

    data = results[0].masks.data
    raw_masks = np.asarray(data.cpu() if hasattr(data, "cpu") else data)  # type: ignore[union-attr]

    if not boxes:
        # Auto-mode: pick the single largest mask as a best-guess for the mouse.
        raw_masks = raw_masks[np.argmax([m.sum() for m in raw_masks])][None]

    out: list[np.ndarray] = []
    for raw in raw_masks:
        if raw.shape != (TARGET_SIZE, TARGET_SIZE):
            raw = cv2.resize(
                raw.astype(np.uint8), (TARGET_SIZE, TARGET_SIZE),
                interpolation=cv2.INTER_NEAREST,
            )
        out.append((raw > 0.5).astype(np.uint8))
    return out


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------

def make_preview(img_bgr: np.ndarray, masks: list[np.ndarray]) -> np.ndarray:
    """Return image with colour-coded mask overlays; original left, overlay right."""
    overlay = img_bgr.copy()
    colours = [(0, 200, 0), (0, 120, 255), (255, 80, 0), (180, 0, 255), (0, 220, 180)]
    for i, mask in enumerate(masks):
        c = colours[i % len(colours)]
        overlay[mask > 0] = (overlay[mask > 0] * 0.5 + np.array(c) * 0.5).astype(np.uint8)
    return np.hstack([img_bgr, overlay])


# ---------------------------------------------------------------------------
# Interactive review
# ---------------------------------------------------------------------------

class _BboxEditor:
    """Stateful mouse callback for multi-instance bounding box editing."""

    WIN = "SAM Review"
    # BGR colours for up to 6 simultaneous instances
    COLOURS = [
        (0, 220, 255),   # yellow  – SAM suggestion
        (0, 255, 80),    # green
        (255, 100, 0),   # blue
        (0, 100, 255),   # orange
        (200, 0, 255),   # purple
        (0, 220, 180),   # teal
    ]

    def __init__(self) -> None:
        self._boxes_px: list[list[int]] = []   # confirmed pixel [x1,y1,x2,y2]
        self._drawing = False
        self._p1: tuple[int, int] | None = None
        self._p2: tuple[int, int] | None = None
        self.modified = False  # True if user changed anything

    def reset(self, initial: list[list[int]] | None = None) -> None:
        self._boxes_px = list(initial) if initial else []
        self._drawing = False
        self._p1 = self._p2 = None
        self.modified = False

    def callback(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drawing = True
            self._p1 = (x, y)
            self._p2 = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self._drawing:
            self._p2 = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self._drawing:
            self._drawing = False
            x1, x2 = sorted([self._p1[0], x])  # type: ignore[index]
            y1, y2 = sorted([self._p1[1], y])   # type: ignore[index]
            if x2 - x1 > 5 and y2 - y1 > 5:
                self._boxes_px.append([x1, y1, x2, y2])
                self.modified = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right-click: remove the innermost box whose rect contains the cursor.
            for i in range(len(self._boxes_px) - 1, -1, -1):
                bx1, by1, bx2, by2 = self._boxes_px[i]
                if bx1 <= x <= bx2 and by1 <= y <= by2:
                    self._boxes_px.pop(i)
                    self.modified = True
                    break

    def render(
        self,
        img: np.ndarray,
        masks: list[np.ndarray],
        title: str,
    ) -> np.ndarray:
        canvas = img.copy()
        h, w = canvas.shape[:2]

        # Colour-coded mask overlays
        for i, mask in enumerate(masks):
            c = self.COLOURS[i % len(self.COLOURS)]
            layer = np.zeros_like(canvas)
            layer[mask > 0] = c
            canvas = cv2.addWeighted(canvas, 0.65, layer, 0.35, 0)

        # Confirmed boxes (numbered)
        for i, (bx1, by1, bx2, by2) in enumerate(self._boxes_px):
            c = self.COLOURS[i % len(self.COLOURS)]
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), c, 2)
            cv2.putText(canvas, str(i + 1), (bx1 + 4, by1 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)

        # Live drag rectangle
        if self._drawing and self._p1 and self._p2:
            cv2.rectangle(canvas, self._p1, self._p2, (255, 255, 255), 2)

        # Instruction bar
        bar = np.zeros((36, w, 3), dtype=np.uint8)
        n = len(self._boxes_px)
        cv2.putText(
            bar,
            f"{title}  [{n} box{'es' if n != 1 else ''}]   Enter=confirm  S=skip  drag=add  RClick=remove  D=del last",
            (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (210, 210, 210), 1, cv2.LINE_AA,
        )
        return np.vstack([bar, canvas])

    def boxes_normalized(self, img_w: int, img_h: int) -> list[list[float]]:
        out: list[list[float]] = []
        for x1, y1, x2, y2 in self._boxes_px:
            out.append([
                (x1 + x2) / 2 / img_w,
                (y1 + y2) / 2 / img_h,
                (x2 - x1) / img_w,
                (y2 - y1) / img_h,
            ])
        return out

    def boxes_xyxy(self) -> list[list[float]]:
        return [[float(v) for v in b] for b in self._boxes_px]


def review_negative(
    editor: "_BboxEditor",
    img: np.ndarray,
    title: str,
) -> bool:
    """
    Show image for negative (background) confirmation.
    Enter = confirm as background (empty label).
    S = skip (do not include in dataset).
    Returns True to keep, False to skip.
    """
    editor.reset()
    h, w = img.shape[:2]
    cv2.namedWindow(editor.WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(editor.WIN, 700, 740)
    cv2.setMouseCallback(editor.WIN, lambda *a: None)  # no drawing needed

    while True:
        canvas = img.copy()
        bar = np.zeros((36, w, 3), dtype=np.uint8)
        cv2.putText(
            bar,
            f"{title}   Enter=background(keep)   S=skip",
            (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (210, 210, 210), 1, cv2.LINE_AA,
        )
        cv2.imshow(editor.WIN, np.vstack([bar, canvas]))
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10):
            return True
        if key in (ord('s'), ord('S')):
            return False


def review_label(
    editor: _BboxEditor,
    sam: SAM,
    img: np.ndarray,
    masks: list[np.ndarray],
    sam_boxes_norm: list[list[float]],
    label_format: str,
    title: str,
) -> tuple[list[np.ndarray], list[list[float]]]:
    """
    Show image + all SAM masks/boxes.  User can add/remove boxes.

    Left drag  = add a new box
    Right-click inside a box = remove it
    D key      = remove last box
    Enter      = confirm (re-runs SAM if boxes were changed)
    S key      = skip this image (returns empty lists)
    """
    h, w = img.shape[:2]

    # Convert normalised SAM boxes to pixel coords for the editor.
    def _to_px(norm: list[float]) -> list[int]:
        cx, cy, bw, bh = norm
        return [
            int((cx - bw / 2) * w), int((cy - bh / 2) * h),
            int((cx + bw / 2) * w), int((cy + bh / 2) * h),
        ]

    editor.reset(initial=[_to_px(b) for b in sam_boxes_norm])

    cv2.namedWindow(editor.WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(editor.WIN, 700, 740)
    cv2.setMouseCallback(editor.WIN, editor.callback)

    active_masks = list(masks)

    while True:
        frame = editor.render(img, active_masks, title)
        cv2.imshow(editor.WIN, frame)
        key = cv2.waitKey(20) & 0xFF

        if key in (ord('d'), ord('D')) and editor._boxes_px:
            editor._boxes_px.pop()
            editor.modified = True

        elif key in (13, 10):  # Enter — confirm
            if editor.modified:
                # Re-run SAM with current boxes for fresh masks.
                current_xyxy = editor.boxes_xyxy()
                if current_xyxy:
                    active_masks = get_masks_for_boxes(sam, img, current_xyxy)
                else:
                    active_masks = []
            break

        elif key in (ord('s'), ord('S')):
            return [], []

    return active_masks, editor.boxes_normalized(w, h)


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
    dataset_type: str,
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
    print(f"Dataset : {dataset_type}")
    print(f"Total images: {len(paths)}, grayscale: {n_gray}")

    # Negative dataset — no YOLO/SAM needed.
    if dataset_type == "negative":
        editor = _BboxEditor() if interactive else None
        if interactive:
            print("Interactive mode: Enter=keep as background   S=skip")
        kept = skipped_neg = 0
        for idx, path in enumerate(paths):
            img_bgr = cv2.imread(str(path))
            if img_bgr is None:
                skipped_neg += 1
                continue
            img_bgr = cv2.resize(img_bgr, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LINEAR)
            if idx in gray_indices:
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                img_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            if interactive and editor is not None:
                title = f"{idx + 1}/{len(paths)}  {path.name}"
                if not review_negative(editor, img_bgr, title):
                    skipped_neg += 1
                    continue
            stem = f"{path.stem}_{idx}"
            cv2.imwrite(str(images_out / f"{stem}.jpg"), img_bgr)
            (labels_out / f"{stem}.txt").write_text("")  # empty = background
            kept += 1
            if (idx + 1) % 20 == 0 or (idx + 1) == len(paths):
                print(f"  [{idx+1:>4}/{len(paths)}] kept={kept} skipped={skipped_neg}")
        if interactive:
            cv2.destroyAllWindows()
        yaml_path = out_dir / "dataset.yaml"
        yaml_path.write_text(textwrap.dedent(f"""\
            path: {out_dir.resolve().as_posix()}
            train: images
            val: images

            nc: 1
            names:
              0: mouse
        """))
        print(f"\nDone (negative).")
        print(f"  Kept     : {kept}")
        print(f"  Skipped  : {skipped_neg}")
        print(f"  Images   \u2192 {images_out}")
        print(f"  Labels   \u2192 {labels_out}  (all empty = background)")
        print(f"  YAML     \u2192 {yaml_path}")
        return

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

        # 1. Detect all mice with YOLO for SAM box prompts.
        yolo_boxes = detect_mouse_boxes(yolo, img_bgr)

        # 2. Segment with SAM (one mask per box, or auto-mode if none found).
        masks = get_masks_for_boxes(sam, img_bgr, yolo_boxes)

        # 3. Derive normalised bboxes from SAM masks.
        sam_boxes_norm: list[list[float]] = []
        for m in masks:
            bb = mask_to_bbox(m)
            if bb:
                sam_boxes_norm.append(bb)

        # 4. Interactive review (or use SAM result directly).
        if interactive and editor is not None:
            title = f"{idx + 1}/{len(paths)}  {path.name}"
            masks, sam_boxes_norm = review_label(
                editor, sam, img_bgr, masks, sam_boxes_norm, label_format, title
            )

        # 5. Build label lines (one per detected mouse).
        label_lines: list[str] = []
        if label_format == "seg":
            for m in masks:
                pts = mask_to_polygon(m)
                if pts:
                    coords = " ".join(f"{v:.6f}" for v in pts)
                    label_lines.append(f"{YOLO_CLASS_ID} {coords}")
        else:
            for bb in sam_boxes_norm:
                coords = " ".join(f"{v:.6f}" for v in bb)
                label_lines.append(f"{YOLO_CLASS_ID} {coords}")

        # 6. Save image.
        stem = f"{path.stem}_{idx}"
        out_img = images_out / f"{stem}.jpg"
        cv2.imwrite(str(out_img), img_bgr)

        # 7. Save label (one line per mouse; empty file = no mouse found).
        label_path = labels_out / f"{stem}.txt"
        label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""))

        if label_lines:
            labeled += 1
        else:
            empty += 1

        # 8. Optional preview.
        if preview:
            prev = make_preview(img_bgr, masks)
            cv2.imwrite(str(previews_out / f"{stem}.jpg"), prev)

        if (idx + 1) % 20 == 0 or (idx + 1) == len(paths):
            status = "gray" if idx in gray_indices else "color"
            box_tag = f"YOLO({len(yolo_boxes)})+SAM" if yolo_boxes else "SAM-auto"
            n_inst = len(label_lines)
            print(f"  [{idx+1:>4}/{len(paths)}] {path.name} | {status} | {box_tag} | instances={n_inst} | labeled={labeled}")

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
    parser.add_argument(
        "--dataset",
        choices=["positive", "negative"],
        default="positive",
        help=(
            "'positive': mouse images — runs SAM and writes YOLO bbox/seg labels (default). "
            "'negative': background images — writes empty .txt labels for YOLO hard-negative training."
        ),
    )
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
        dataset_type=args.dataset,
    )


if __name__ == "__main__":
    print("GPU available:", torch.cuda.is_available())
    main()
