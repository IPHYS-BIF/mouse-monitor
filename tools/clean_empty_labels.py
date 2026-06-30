"""Delete images whose YOLO label file is empty (skipped during labeling).

Scans --labels-dir for .txt files that are empty (zero bytes or whitespace
only) and removes both the label file and the matching image from --images-dir.

Usage
-----
    # dry-run first (no files deleted)
    python tools/clean_empty_labels.py --labeled-dir data/labeled/positive

    # actually delete
    python tools/clean_empty_labels.py --labeled-dir data/labeled/positive --delete
"""

import argparse
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_empty_pairs(labels_dir: Path, images_dir: Path) -> list[tuple[Path, Path | None]]:
    """Return (label_path, image_path_or_None) for every empty label file."""
    pairs: list[tuple[Path, Path | None]] = []
    for label_path in sorted(labels_dir.glob("*.txt")):
        if label_path.read_text().strip():
            continue  # has content → keep
        # Find matching image (any supported extension)
        img_path: Path | None = None
        for ext in IMAGE_EXTENSIONS:
            candidate = images_dir / (label_path.stem + ext)
            if candidate.exists():
                img_path = candidate
                break
        pairs.append((label_path, img_path))
    return pairs


def clean(labeled_dir: Path, delete: bool) -> None:
    labels_dir = labeled_dir / "labels"
    images_dir = labeled_dir / "images"

    if not labels_dir.exists():
        print(f"Labels folder not found: {labels_dir}")
        return
    if not images_dir.exists():
        print(f"Images folder not found: {images_dir}")
        return

    pairs = find_empty_pairs(labels_dir, images_dir)

    if not pairs:
        print("No empty label files found — nothing to clean.")
        return

    action = "Deleting" if delete else "Would delete (dry-run)"
    removed_labels = removed_images = 0

    for label_path, img_path in pairs:
        img_tag = img_path.name if img_path else "(no matching image)"
        print(f"  {action}: {label_path.name}  +  {img_tag}")
        if delete:
            label_path.unlink()
            removed_labels += 1
            if img_path and img_path.exists():
                img_path.unlink()
                removed_images += 1

    if delete:
        print(f"\nRemoved {removed_labels} label files and {removed_images} image files.")
    else:
        print(f"\nDry-run: {len(pairs)} pair(s) would be removed.")
        print("Re-run with --delete to actually remove them.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove images + empty label files from a YOLO labeled dataset folder."
    )
    parser.add_argument(
        "--labeled-dir",
        default="data/labeled/positive",
        help="Root folder containing images/ and labels/ sub-folders (default: data/labeled/positive).",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete files. Without this flag the script only prints what would be removed.",
    )
    args = parser.parse_args()
    clean(Path(args.labeled_dir), delete=args.delete)


if __name__ == "__main__":
    main()
