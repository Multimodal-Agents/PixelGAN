#!/usr/bin/env python3
"""
Tree Dataset Generator — Procedural pixel art trees for all size tiers.

Generates diverse tree sprite datasets using 15 tree types × 5 seasons across
all size categories (8×8, 16×16, 32×32, 64×64), automatically applying the
correct dithering profile for each size.

Tree types:
  pine, maple, birch, oak, dead, generic   — original 6
  willow, spruce, cherry, acacia, shrub    — 5 new (weeping/monopodial/spiral/flat/multi-stem)
  palm, fir, apple, cypress                — 4 more (crown/herringbone/dotted/dense-needle)

Branching algorithms:
  recursive    — symmetric balanced tree (pine, maple, birch, oak, apple, cypress, acacia)
  monopodial   — dominant central axis with regular side branches (spruce)
  weeping      — progressive gravity droop per depth level (willow)
  herringbone  — alternating left/right single branches (fir)
  spiral       — n-way fan with per-depth rotation offset (cherry)
  multi_stem   — multiple trunks from base (shrub)
  crown        — radial frond fan from trunk top (palm)

Dithering per size (SIZE_DITHER_PROFILES):
  8×8   → none         (pixel art is too small for dithering)
  16×16 → bayer2x2     (light ordered dithering)
  32×32 → bayer4x4     (standard pixel art look)
  64×64 → atkinson     (smooth Mac-classic style)

Usage:
    # Generate ALL size tiers at once (recommended):
    python scripts/generate_trees.py --all-sizes

    # Single size with default settings:
    python scripts/generate_trees.py --size 64

    # Control sample count and output:
    python scripts/generate_trees.py --all-sizes --n-per-type 40 --output datasets/trees

    # Preview sheet only (no parquet):
    python scripts/generate_trees.py --size 32 --preview-only

    # Text-captioned dataset (for text-conditioned GAN training):
    python scripts/generate_trees.py --all-sizes --dataset-mode text

    # Single tree type debug:
    python scripts/generate_trees.py --size 64 --type willow --n-per-type 8 --preview-only
"""

import sys
import argparse
import math
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pixelgan.data.tree_generator import (
    ProceduralTreeGenerator,
    generate_tree_batch,
    generate_all_tree_batches,
    TREE_TYPES,
)
from pixelgan.data.color_palette import TREE_PALETTE_CONFIGS, generate_tree_palette
from pixelgan.data.dithering import get_size_dither_profile
from pixelgan.training.dataset import create_seed_dataset, create_text_dataset


# All palette types covered across all tree categories
ALL_PALETTE_TYPES = list(TREE_PALETTE_CONFIGS.keys())

# ---------------------------------------------------------------------------
# Retro platform presets
# The GAN always trains at the canvas pixel size (always square).
# Pick the platform that matches the game you're making; the canvas size IS
# the GAN input/output resolution — no separate "GAN size" vs "art size".
# ---------------------------------------------------------------------------
RETRO_PLATFORMS: dict[str, int] = {
    "nes":        8,    # 8×8  NES / Game Boy tile
    "gb":         8,    # alias
    "gameboy":    8,    # alias
    "snes_small": 16,   # 16×16 SNES small sprite
    "snes":       32,   # 32×32 SNES / 16-bit large sprite (most common)
    "gba":        64,   # 64×64 GBA / N64 era
    "hd_retro":  128,   # 128×128 modern retro-style
}

# Size tiers with appropriate display scale for preview sheets
SIZE_TIERS = {
    8:   {"display_scale": 8, "label": "8×8",    "platform": "NES/GB tile"},
    16:  {"display_scale": 4, "label": "16×16",   "platform": "SNES small sprite"},
    32:  {"display_scale": 2, "label": "32×32",   "platform": "SNES/16-bit sprite"},
    64:  {"display_scale": 1, "label": "64×64",   "platform": "GBA/N64 sprite"},
    128: {"display_scale": 1, "label": "128×128", "platform": "HD retro sprite"},
}


def make_preview_sheet(
    gen: ProceduralTreeGenerator,
    size: int,
    output_dir: Path,
    palette_types: list[str],
    seeds_per_type: int = 4,
    display_scale: int = 1,
) -> Path:
    """
    Render a preview sheet showing all tree types at the given size.

    Grid layout: one row per palette_type, seeds_per_type columns of variations.
    """
    from PIL import Image
    import numpy as np

    profile = get_size_dither_profile(size)
    print(f"  Building preview sheet: {len(palette_types)} types × {seeds_per_type} seeds "
          f"| dither={profile['method']}")

    sprite_px = size * display_scale
    pad = max(2, display_scale)
    cols = seeds_per_type
    rows = len(palette_types)

    sheet_w = cols * sprite_px + (cols + 1) * pad
    sheet_h = rows * sprite_px + (rows + 1) * pad
    sheet = Image.new("RGBA", (sheet_w, sheet_h), (18, 18, 28, 255))

    for row_idx, palette_type in enumerate(palette_types):
        cfg = TREE_PALETTE_CONFIGS.get(palette_type)
        tree_type, season = gen.PALETTE_TO_TREE.get(palette_type, ("generic", "summer"))

        for col_idx in range(seeds_per_type):
            seed = col_idx * 31 + row_idx * 7
            canvas = gen.generate(tree_type, size=size, seed=seed, season=season)
            pal_seed = seed + 500
            img = gen.render_to_image(canvas, palette_type, seed=pal_seed)

            if display_scale > 1:
                img = img.resize((sprite_px, sprite_px), resample=Image.NEAREST)

            x = pad + col_idx * (sprite_px + pad)
            y = pad + row_idx * (sprite_px + pad)
            sheet.paste(img, (x, y), img)

    out_path = output_dir / f"preview_trees_{size}x{size}.png"
    sheet.save(out_path)
    print(f"  Preview saved → {out_path}")
    return out_path


def generate_for_size(
    size: int,
    n_per_type: int,
    output_dir: Path,
    dataset_mode: str,
    base_seed: int,
    preview_only: bool,
    type_filter: str | None,
) -> None:
    """Generate tree sprites and dataset for a single size tier."""
    profile = get_size_dither_profile(size)
    dither_method = profile["method"]
    tier = SIZE_TIERS.get(size, {"display_scale": 1, "label": f"{size}x{size}", "platform": "custom"})
    display_scale = tier["display_scale"]
    label = tier["label"]

    print(f"\n{'='*60}")
    platform = tier.get("platform", f"{size}×{size}")
    print(f"  Size tier: {label}  |  platform: {platform}")
    print(f"  Canvas:    {size}×{size} px  (GAN trains at this exact resolution — always square)")
    print(f"  Dither:    {dither_method} (intensity={profile['intensity']:.2f})")
    print(f"  Samples/type: {n_per_type}  (×2 with flips)")
    print(f"{'='*60}")

    gen = ProceduralTreeGenerator()

    # Choose palette types to generate
    if type_filter:
        # Filter by tree type name
        palette_types = [
            pt for pt in ALL_PALETTE_TYPES
            if gen.PALETTE_TO_TREE.get(pt, ("", ""))[0] == type_filter
        ]
        if not palette_types:
            print(f"  Warning: no palette types found for tree type '{type_filter}'")
            palette_types = ALL_PALETTE_TYPES
    else:
        palette_types = ALL_PALETTE_TYPES

    print(f"  Tree categories: {len(palette_types)} palette types")
    print(f"  Types: {', '.join(sorted(set(gen.PALETTE_TO_TREE.get(pt, ('?',''))[0] for pt in palette_types)))}")

    # Preview sheet
    make_preview_sheet(
        gen, size, output_dir, palette_types,
        seeds_per_type=min(6, n_per_type),
        display_scale=display_scale,
    )

    if preview_only:
        return

    # Generate training samples
    print(f"\n  Generating training samples...")
    all_samples = []
    for palette_type in palette_types:
        batch = generate_tree_batch(
            palette_type,
            size=size,
            n=n_per_type,
            base_seed=base_seed,
            include_flipped=True,
            display_scale=1,  # store at native res, not scaled
        )
        all_samples.extend(batch)

    total = len(all_samples)
    print(f"  Total samples generated: {total}")

    # Save parquet dataset
    if dataset_mode == "seed":
        image_bytes = [s["image_bytes"] for s in all_samples]
        seeds = [s["seed"] for s in all_samples]
        out_path = output_dir / f"sprites_seed_{size}x{size}_trees.parquet"
        create_seed_dataset(image_bytes, seeds, str(out_path))
        print(f"  Seed dataset saved → {out_path}")
        print(f"  Rows: {total}  |  Format: [seed, image_bytes]")
    elif dataset_mode == "text":
        image_bytes = [s["image_bytes"] for s in all_samples]
        captions = [s["caption"] for s in all_samples]
        out_path = output_dir / f"sprites_text_{size}x{size}_trees.parquet"
        create_text_dataset(image_bytes, captions, str(out_path))
        print(f"  Text dataset saved → {out_path}")
        print(f"  Rows: {total}  |  Format: [caption, image_bytes]")


def cmd_generate(args) -> None:
    """Main generation command handler."""
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    # --all-sizes covers the four standard retro tiers (not the HD 128 tier)
    ALL_STANDARD_SIZES = [8, 16, 32, 64]
    sizes = ALL_STANDARD_SIZES if args.all_sizes else [args.size]

    print(f"\nPixelGAN Tree Dataset Generator")
    sizes_display = [f"{s}x{s} ({SIZE_TIERS.get(s, {}).get('platform', '')})" for s in sizes]
    print(f"  Output:       {output}")
    print(f"  Sizes:        {sizes_display}")
    print(f"  N per type:   {args.n_per_type}")
    print(f"  Dataset mode: {args.dataset_mode}")
    print(f"  Preview only: {args.preview_only}")
    if args.type:
        print(f"  Tree filter:  {args.type}")

    for size in sizes:
        generate_for_size(
            size=size,
            n_per_type=args.n_per_type,
            output_dir=output,
            dataset_mode=args.dataset_mode,
            base_seed=args.seed,
            preview_only=args.preview_only,
            type_filter=args.type,
        )

    print(f"\n{'='*60}")
    print(f"  Done! Output: {output}")
    if not args.preview_only:
        datasets = list(output.glob("sprites_*.parquet"))
        if datasets:
            total_bytes = sum(p.stat().st_size for p in datasets)
            print(f"  Parquet files: {len(datasets)}")
            print(f"  Total size:    {total_bytes / 1024 / 1024:.1f} MB")
    print(f"{'='*60}")


def cmd_list(args) -> None:
    """List all available tree types and palette configs."""
    gen = ProceduralTreeGenerator()

    print(f"\n{'='*60}")
    print(f" Available tree types ({len(gen.PRESETS)})")
    print(f"{'='*60}")
    for tree_type, preset in sorted(gen.PRESETS.items()):
        algo = preset.get("branching_algo", "recursive")
        leaf = preset.get("leaf_type", "?")
        trunk = preset.get("trunk_frac", 0)
        print(f"  {tree_type:<12}  algo={algo:<14} leaf={leaf:<14} trunk={trunk:.0%}")

    print(f"\n{'='*60}")
    print(f" Palette configs ({len(ALL_PALETTE_TYPES)})")
    print(f"{'='*60}")
    for pt in ALL_PALETTE_TYPES:
        tree_type, season = gen.PALETTE_TO_TREE.get(pt, ("?", "?"))
        print(f"  {pt:<22}  → {tree_type:<10} {season}")

    print(f"\n{'='*60}")
    print(f" Dithering profiles per size")
    print(f"{'='*60}")
    for size in [8, 16, 32, 64, 128]:
        p = get_size_dither_profile(size)
        tier = SIZE_TIERS.get(size, {})
        plat = tier.get("platform", "")
        print(f"  {size:>4}×{size:<4}  {plat:<24}  method={p['method']:<18} intensity={p['intensity']:.2f}")

    print(f"\n{'='*60}")
    print(f" Retro platform presets  (--platform NAME  →  --size N)")
    print(f"{'='*60}")
    for name in sorted(RETRO_PLATFORMS):
        sz = RETRO_PLATFORMS[name]
        tier = SIZE_TIERS.get(sz, {})
        plat = tier.get("platform", "")
        print(f"  --platform {name:<14}  →  {sz}×{sz}  ({plat})")


def main():
    parser = argparse.ArgumentParser(
        description="PixelGAN Procedural Tree Dataset Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    # ---- generate (default) ----
    gen_p = sub.add_parser("generate", help="Generate tree datasets")
    gen_p.add_argument("--all-sizes", action="store_true",
                       help="Generate all size tiers: 8, 16, 32, 64")
    gen_p.add_argument("--size", type=int, default=None, choices=[8, 16, 32, 64, 128],
                       help="Canvas size in pixels (always square). Ignored if --all-sizes or --platform.")
    gen_p.add_argument("--platform", default=None, choices=sorted(RETRO_PLATFORMS.keys()),
                       help="Retro platform preset (sets canvas size). E.g. snes=32×32, gba=64×64")
    gen_p.add_argument("--n-per-type", type=int, default=20,
                       help="Samples per palette type before flips (default: 20)")
    gen_p.add_argument("--output", default="datasets/sprites",
                       help="Output directory (default: datasets/sprites)")
    gen_p.add_argument("--dataset-mode", choices=["seed", "text"], default="seed",
                       help="Dataset format: seed (unconditional) or text (captioned)")
    gen_p.add_argument("--preview-only", action="store_true",
                       help="Only render preview sheets, skip parquet")
    gen_p.add_argument("--type", default=None,
                       help="Filter to a single tree type (e.g. willow, palm, fir)")
    gen_p.add_argument("--seed", type=int, default=42,
                       help="Base random seed")

    # ---- list ----
    sub.add_parser("list", help="List all tree types, palettes, and dither profiles")

    # ---- top-level shortcuts (no subcommand needed) ----
    parser.add_argument("--all-sizes", action="store_true",
                        help="Generate all size tiers (shortcut, no subcommand needed)")
    parser.add_argument("--size", type=int, default=None, choices=[8, 16, 32, 64, 128],
                        help="Canvas size in pixels (always square). GAN trains at this exact size.")
    parser.add_argument("--platform", default=None, choices=sorted(RETRO_PLATFORMS.keys()),
                        help=f"Retro platform preset → canvas size. Options: {', '.join(f'{k}={v}px' for k,v in sorted(RETRO_PLATFORMS.items()))}")
    parser.add_argument("--n-per-type", type=int, default=20)
    parser.add_argument("--output", default="datasets/sprites")
    parser.add_argument("--dataset-mode", choices=["seed", "text"], default="seed")
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--type", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--list", action="store_true", help="List available types")

    args = parser.parse_args()

    # Resolve --platform to a size
    if hasattr(args, "platform") and args.platform and args.platform in RETRO_PLATFORMS:
        args.size = RETRO_PLATFORMS[args.platform]
    elif args.size is None and not getattr(args, "all_sizes", False):
        args.size = 32  # sensible default: SNES-style

    if args.command == "list" or getattr(args, "list", False):
        cmd_list(args)
    else:
        cmd_generate(args)


if __name__ == "__main__":
    main()
