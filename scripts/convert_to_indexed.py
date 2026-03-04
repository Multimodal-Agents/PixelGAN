#!/usr/bin/env python3
"""
convert_to_indexed.py — Option B: Convert PNG-parquet to indexed-palette-parquet.

Takes an existing seed->image parquet dataset where the "image" column holds
raw PNG bytes and writes a new parquet file with the indexed palette format:

    Input columns:    seed (int64), image (bytes PNG)
    Output columns:   seed (int64), index_map (bytes), palette_data (bytes),
                      n_colors (int64)

The indexed format stores each sprite as:
  - A uint8 [H, W] index map (one byte per pixel → palette index)
  - A uint8 [N, 3] RGB palette (N ≤ max_colors)

Benefits over storing raw PNG:
  - ~10–50× faster data loading at training time (no PIL decode, no colour ops)
  - Transparent pixel is explicitly encoded as palette index 0 = background
  - Palette can be passed directly to the Generator's PaletteLookup (Option A)
    so the GAN learns to stay within the sprite's natural colour range
  - Enables "palette coherence loss" during training

Usage:
    # Convert a single parquet file:
    python scripts/convert_to_indexed.py \
        --input  datasets/sprites/trees.parquet \
        --output datasets/sprites/trees_indexed.parquet \
        --max-colors 8

    # Convert all parquets in a directory:
    python scripts/convert_to_indexed.py \
        --input  datasets/sprites/ \
        --output datasets/sprites_indexed/ \
        --max-colors 8 \
        --image-size 64

    # Preview statistics only (dry run):
    python scripts/convert_to_indexed.py \
        --input datasets/sprites/trees.parquet \
        --stats-only
"""

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert PNG-parquet dataset to indexed palette format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input", "-i", required=True,
        help="Path to input .parquet file or directory of .parquet files.",
    )
    p.add_argument(
        "--output", "-o", default=None,
        help=(
            "Output path. If omitted, appends '_indexed' to the input name. "
            "Use a directory path when --input is also a directory."
        ),
    )
    p.add_argument(
        "--max-colors", type=int, default=8,
        help="Maximum palette size per sprite (default: 8). "
             "Sprites with more colours will be quantized via median-cut.",
    )
    p.add_argument(
        "--image-size", type=int, default=None,
        help="Resize sprites to this size before indexing (default: keep original).",
    )
    p.add_argument(
        "--stats-only", action="store_true",
        help="Print palette statistics without writing output files.",
    )
    p.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress progress output.",
    )
    return p.parse_args()


def _infer_output_path(input_path: Path) -> Path:
    """Derive output path from input path."""
    if input_path.is_dir():
        return input_path.parent / (input_path.name + "_indexed")
    else:
        stem = input_path.stem
        return input_path.parent / (stem + "_indexed.parquet")


def convert_file(
    input_path: Path,
    output_path: Path,
    max_colors: int,
    image_size: int | None,
    stats_only: bool,
    quiet: bool,
    global_palette: bool = False,
    auto_k: bool = False,
) -> dict:
    """
    Convert a single parquet file.

    Returns:
        stats dict with keys: n_sprites, mean_colors, max_colors_seen,
                               mean_unique_colors, n_quantized
    """
    import io
    import numpy as np
    import pandas as pd
    from PIL import Image

    # Late import so the script is importable without pyarrow
    try:
        from pixelgan.data.indexed_format import (
            rgba_to_indexed,
            save_indexed_parquet,
            IndexedSprite,
            build_global_palette,
            remap_to_global_palette,
            measure_color_diversity,
        )
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from pixelgan.data.indexed_format import (
            rgba_to_indexed,
            save_indexed_parquet,
            IndexedSprite,
            build_global_palette,
            remap_to_global_palette,
            measure_color_diversity,
        )

    if not quiet:
        print(f"\n  Reading {input_path} ...", flush=True)

    df = pd.read_parquet(input_path, columns=["seed", "image"])
    n = len(df)

    # ── Phase 1: decode all sprites (always needed) ──────────────────────
    sprites: list[IndexedSprite] = []
    seeds:   list[int]           = []
    n_quantized = 0
    color_counts: list[int] = []

    for i, row in df.iterrows():
        img = Image.open(io.BytesIO(bytes(row["image"]))).convert("RGBA")
        if image_size is not None and img.size != (image_size, image_size):
            img = img.resize((image_size, image_size), Image.NEAREST)

        arr = np.array(img)
        unique_before = len(set(map(tuple, arr.reshape(-1, 4).tolist())))
        color_counts.append(unique_before)

        sprite = rgba_to_indexed(img, max_colors=max_colors)
        if sprite.n_colors < unique_before:
            n_quantized += 1

        sprites.append(sprite)
        seeds.append(int(row["seed"]))

        if not quiet and (i % 200 == 0 or i == n - 1):
            print(f"  [Phase 1] [{i + 1:>5}/{n}] unique_colors={unique_before}",
                  end="\r", flush=True)

    if not quiet:
        print()

    # ── Phase 2 (global palette): measure diversity → k-means → remap ────
    chosen_k = max_colors
    if global_palette:
        if not quiet:
            print("  Measuring dataset colour diversity...", flush=True)
        diversity = measure_color_diversity(sprites, max_k=max_colors)
        n_unique  = diversity["n_unique_colors"]

        if auto_k:
            chosen_k = max(1, diversity["recommended_k"])
            if not quiet:
                print(f"  Unique colours in dataset : {n_unique:,}")
                print(f"  Auto-selected palette size: {chosen_k}  "
                      f"(elbow of k-means inertia, capped at {max_colors})")
        else:
            chosen_k = min(max_colors, n_unique)
            if not quiet:
                print(f"  Unique colours in dataset : {n_unique:,}")
                print(f"  Using palette size        : {chosen_k}")

        if not quiet:
            print(f"  Running k-means (k={chosen_k}) over {n_unique:,} unique colours...",
                  flush=True)

        shared_pal = build_global_palette(sprites, n_colors=chosen_k)

        if not quiet:
            print(f"  Remapping {n:,} sprites to global palette...", flush=True)
        sprites = [
            remap_to_global_palette(s, shared_pal) for s in sprites
        ]
        if not quiet:
            print(f"  Global palette built: {chosen_k} visible slots + 1 transparent")

    stats = {
        "n_sprites":          n,
        "mean_unique_colors": float(np.mean(color_counts)),
        "max_colors_seen":    int(np.max(color_counts)),
        "min_colors_seen":    int(np.min(color_counts)),
        "n_quantized":        n_quantized,
        "palette_size":       chosen_k,
        "global_palette":     global_palette,
    }

    if not quiet:
        print(f"  n_sprites         = {stats['n_sprites']:,}")
        print(f"  colours/sprite    = {stats['mean_unique_colors']:.1f} avg, "
              f"{stats['min_colors_seen']} min, {stats['max_colors_seen']} max")
        if not global_palette:
            print(f"  quantized (>{max_colors} colours) = {n_quantized} "
                  f"({100 * n_quantized / n:.1f}%)")
        else:
            print(f"  palette mode      = global shared ({chosen_k} colours)")

    if not stats_only:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_indexed_parquet(sprites, seeds, output_path)
        if not quiet:
            import os
            size_kb = os.path.getsize(output_path) / 1024
            print(f"  Saved → {output_path}  ({size_kb:.0f} KB)")

    return stats


def main() -> None:
    args = parse_args()
    input_path  = Path(args.input).resolve()
    output_path = Path(args.output).resolve() if args.output else None
    gp = getattr(args, "global_palette", False)
    ak = getattr(args, "auto_k", False)

    if input_path.is_dir():
        parquet_files = sorted(input_path.glob("*.parquet"))
        if not parquet_files:
            print(f"ERROR: No .parquet files found in {input_path}", file=sys.stderr)
            sys.exit(1)

        if output_path is None:
            output_path = _infer_output_path(input_path, global_palette=gp)

        if not args.quiet:
            print(f"Converting {len(parquet_files)} parquet file(s)")
            print(f"  input_dir      = {input_path}")
            print(f"  output_dir     = {output_path}")
            print(f"  max_colors     = {args.max_colors}")
            print(f"  global_palette = {gp}  (auto_k={ak})")
            if args.image_size:
                print(f"  image_size  = {args.image_size}")
            if args.stats_only:
                print("  (stats-only mode — no output files written)")

        all_stats: list[dict] = []
        for f in parquet_files:
            out_f = output_path / f.name.replace(".parquet", "_indexed.parquet")
            stats = convert_file(
                f, out_f,
                max_colors=args.max_colors,
                image_size=args.image_size,
                stats_only=args.stats_only,
                quiet=args.quiet,
                global_palette=gp,
                auto_k=ak,
            )
            all_stats.append(stats)

        if not args.quiet:
            import numpy as np
            total = sum(s["n_sprites"] for s in all_stats)
            total_q = sum(s["n_quantized"] for s in all_stats)
            print(f"\nDone — {total:,} sprites total, "
                  f"{total_q} quantized ({100 * total_q / total:.1f}%)")

    elif input_path.is_file():
        if output_path is None:
            output_path = _infer_output_path(input_path, global_palette=gp)

        if not args.quiet:
            print(f"Converting {input_path.name}")
            print(f"  input          = {input_path}")
            print(f"  output         = {output_path}")
            print(f"  max_colors     = {args.max_colors}")
            print(f"  global_palette = {gp}  (auto_k={ak})")
            if args.image_size:
                print(f"  image_size  = {args.image_size}")
            if args.stats_only:
                print("  (stats-only mode — no output files written)")

        convert_file(
            input_path, output_path,
            max_colors=args.max_colors,
            image_size=args.image_size,
            stats_only=args.stats_only,
            quiet=args.quiet,
            global_palette=gp,
            auto_k=ak,
        )
        print("\nDone.")

    else:
        print(f"ERROR: Input path not found: {input_path}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
