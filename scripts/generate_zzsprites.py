#!/usr/bin/env python3
"""
ZzSprite Dataset Generator — procedural pixel art sprites.

Ports ZzSprite.js (Frank Force, MIT) to Python and generates training datasets
for PixelGAN. Produces organic-looking symmetric sprites with genuine alpha
transparency — a complementary data source alongside procedural trees.

Four colour modes:
  0 = colored    — random HSL with black outline  (alien/creature style)
  1 = grayscale  — black / gray / white            (rock/metal/bone items)
  2 = silhouette — solid white fill, black outline (icon / power-up style)
  3 = black      — three-pass solid black stencil  (UI icon style)

Usage:
    # Quick: 64×64 colored sprites (500 total incl. flips)
    python scripts/generate_zzsprites.py --size 64 --n 500

    # All sizes (8, 16, 32, 64)
    python scripts/generate_zzsprites.py --all-sizes --n 300

    # Colored mode only (trains fastest)
    python scripts/generate_zzsprites.py --size 64 --modes 0 --n 600

    # Preview sheet only — no dataset saved
    python scripts/generate_zzsprites.py --size 32 --preview-only

    # Text-captioned format (for text→image conditioning)
    python scripts/generate_zzsprites.py --size 32 --dataset-mode text --n 400
"""

import sys
import io
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pixelgan.data.zzsprite_generator import (
    ZzSpriteGenerator,
    MODE_COLORED, MODE_GRAYSCALE, MODE_SILHOUETTE, MODE_BLACK,
    MODE_NAMES, is_blank,
)
from pixelgan.training.dataset import create_seed_dataset, create_text_dataset


# ---------------------------------------------------------------------------
# Preview sheet
# ---------------------------------------------------------------------------

def make_preview_sheet(
    gen: ZzSpriteGenerator,
    size: int,
    output_dir: Path,
    modes: list,
    seeds_per_row: int = 8,
    display_scale: int = 4,
) -> Path:
    """Render a preview grid: one row per mode, columns = seed variants."""
    from PIL import Image

    pad      = max(2, display_scale // 2)
    cols     = seeds_per_row
    rows     = len(modes)
    sprite_px = size * display_scale

    sheet_w = cols * sprite_px + (cols + 1) * pad
    sheet_h = rows * sprite_px + (rows + 1) * pad
    sheet   = Image.new("RGBA", (sheet_w, sheet_h), (18, 18, 28, 255))

    for row_idx, mode in enumerate(modes):
        for col_idx in range(seeds_per_row):
            seed = col_idx * 31 + 1
            img  = gen.generate(
                seed=seed, size=size, mode=mode,
                mutate_seed=col_idx * 7, color_seed=col_idx * 13,
            )
            if display_scale > 1:
                img = img.resize((sprite_px, sprite_px), Image.NEAREST)
            x = pad + col_idx * (sprite_px + pad)
            y = pad + row_idx * (sprite_px + pad)
            sheet.paste(img, (x, y), img)

    out_path = output_dir / f"preview_zzsprite_{size}x{size}.png"
    sheet.save(out_path)
    print(f"  Preview → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Per-size generation
# ---------------------------------------------------------------------------

DISPLAY_SCALES = {8: 8, 16: 4, 32: 2, 64: 2, 128: 1}


def generate_for_size(
    size: int,
    n: int,
    output_dir: Path,
    dataset_mode: str,
    base_seed: int,
    modes: list,
    preview_only: bool,
) -> None:
    gen = ZzSpriteGenerator()

    print(f"\n{'='*60}")
    print(f"  ZzSprite  {size}×{size}   modes=[{', '.join(MODE_NAMES[m] for m in modes)}]")
    print(f"  Target: {n} sprites")
    print(f"{'='*60}")

    make_preview_sheet(
        gen, size, output_dir, modes,
        seeds_per_row=min(8, n // max(len(modes), 1)),
        display_scale=DISPLAY_SCALES.get(size, 2),
    )

    if preview_only:
        return

    # Generate
    samples = gen.generate_batch(
        n=n, size=size, base_seed=base_seed, modes=modes,
        mutate_variants=3, color_variants=2, include_flipped=True,
    )

    # Filter blank / degenerate sprites
    from PIL import Image as _PIL
    kept = []
    for s in samples:
        img = _PIL.open(io.BytesIO(s["image_bytes"]))
        if not is_blank(img):
            kept.append(s)
    if len(kept) < len(samples):
        print(f"  Filtered {len(samples) - len(kept)} blank sprites")
    samples = kept
    print(f"  Generated: {len(samples)} sprites")

    if dataset_mode == "seed":
        out_path = output_dir / f"sprites_zzsprite_{size}x{size}.parquet"
        create_seed_dataset(
            [s["image_bytes"] for s in samples],
            [s["seed"] for s in samples],
            str(out_path),
        )
    elif dataset_mode == "text":
        out_path = output_dir / f"sprites_zzsprite_text_{size}x{size}.parquet"
        create_text_dataset(
            [s["image_bytes"] for s in samples],
            [s["caption"] for s in samples],
            str(out_path),
        )

    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate ZzSprite procedural pixel art training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--size", type=int, default=32,
                        choices=[8, 16, 32, 64, 128],
                        help="Canvas size (default: 32)")
    parser.add_argument("--all-sizes", action="store_true",
                        help="Generate 8, 16, 32, 64 in one run")
    parser.add_argument("--n", type=int, default=300,
                        help="Number of sprites to generate (default: 300)")
    parser.add_argument("--modes", type=int, nargs="+",
                        default=[0, 1, 2, 3], choices=[0, 1, 2, 3],
                        help="Colour modes: 0=colored 1=gray 2=silhouette 3=black")
    parser.add_argument("--dataset-mode", choices=["seed", "text"], default="seed",
                        help="Output format (default: seed)")
    parser.add_argument("--output", default="datasets/sprites",
                        help="Output directory (default: datasets/sprites)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed (default: 42)")
    parser.add_argument("--preview-only", action="store_true",
                        help="Only save preview PNG, skip parquet")

    args   = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    sizes = [8, 16, 32, 64] if args.all_sizes else [args.size]

    print(f"\nZzSprite Dataset Generator")
    print(f"  Output : {output}")
    print(f"  Sizes  : {sizes}")
    print(f"  Modes  : {[MODE_NAMES[m] for m in args.modes]}")
    print(f"  N      : {args.n}")

    for size in sizes:
        generate_for_size(
            size=size, n=args.n, output_dir=output,
            dataset_mode=args.dataset_mode, base_seed=args.seed,
            modes=args.modes, preview_only=args.preview_only,
        )

    print(f"\n{'='*60}")
    print(f"  Done! Output: {output}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
