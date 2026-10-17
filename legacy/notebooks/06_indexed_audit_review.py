"""Create indexed high-loss contact sheets for manual label curation."""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


WORK_DIR = Path.cwd().resolve()
AUDIT_PATH = WORK_DIR / "outputs" / "metrics" / "resnet18_sample_audit.csv"
FIGURES_DIR = WORK_DIR / "outputs" / "figures"
METRICS_DIR = WORK_DIR / "outputs" / "metrics"
TILE_SIZE = (240, 205)
COLUMNS = 5
ROWS_PER_SPLIT = 40


def _font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def main() -> None:
    rows = list(csv.DictReader(AUDIT_PATH.open(newline="")))
    original_fieldnames = list(rows[0].keys())
    selected = []
    for split in ("train", "val", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        split_rows.sort(key=lambda row: float(row["loss"]), reverse=True)
        for index, row in enumerate(split_rows[:ROWS_PER_SPLIT], start=1):
            row = dict(row)
            row["review_id"] = f"{split}-{index:03d}"
            selected.append(row)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with (METRICS_DIR / "resnet18_high_loss_indexed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["review_id", *original_fieldnames])
        writer.writeheader()
        for row in selected:
            writer.writerow({"review_id": row["review_id"], **{key: row[key] for key in original_fieldnames}})

    for split in ("train", "val", "test"):
        split_rows = [row for row in selected if row["split"] == split]
        canvas = Image.new("RGB", (COLUMNS * TILE_SIZE[0], ((len(split_rows) + COLUMNS - 1) // COLUMNS) * TILE_SIZE[1]), "white")
        draw = ImageDraw.Draw(canvas)
        title_font = _font(16)
        label_font = _font(13)
        for position, row in enumerate(split_rows):
            x = (position % COLUMNS) * TILE_SIZE[0]
            y = (position // COLUMNS) * TILE_SIZE[1]
            image = Image.open(row["path"]).convert("RGB")
            image = ImageOps.contain(image, (TILE_SIZE[0] - 8, 155))
            image_x = x + (TILE_SIZE[0] - image.width) // 2
            canvas.paste(image, (image_x, y + 2))
            draw.text((x + 4, y + 160), row["review_id"], fill="black", font=title_font)
            draw.text(
                (x + 4, y + 178),
                f"{row['label']} → {row['prediction']}  loss={float(row['loss']):.1f}",
                fill="black",
                font=label_font,
            )
            draw.rectangle((x, y, x + TILE_SIZE[0] - 1, y + TILE_SIZE[1] - 1), outline="#bbbbbb")
        canvas.save(FIGURES_DIR / f"resnet18_high_loss_{split}_indexed.jpg", quality=92)

    print(f"Wrote {len(selected)} indexed review rows to {METRICS_DIR}")


if __name__ == "__main__":
    main()
