#!/usr/bin/env python3
"""
Sprite Generator — Creates cute pixel art training data.

Generates sprites in Galaga, Zelda, and Pac-Man styles with:
  - Procedural color variations
  - Dithering options
  - PNG export
  - Parquet dataset creation (ready for PixelGAN training)

Usage:
    # Generate sprite sheet preview
    python scripts/generate_sprites.py --mode preview

    # Generate full training dataset
    python scripts/generate_sprites.py --mode dataset --size 32 --n-per-sprite 16

    # Generate single sprite with all palette variations
    python scripts/generate_sprites.py --mode single --sprite galaga_bee

    # Show all available sprites
    python scripts/generate_sprites.py --mode list
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pixelgan.data.sprite_generator import (
    SPRITES, CATEGORIES, SpriteRenderer,
    generate_sprite_sheet, generate_training_batch, list_sprites,
)
from pixelgan.data.color_palette import (
    PaletteGenerator, PALETTES, get_sprite_palette,
    generate_palette_variations, PICO8, SWEETIE16, ENDESGA32,
)
from pixelgan.training.dataset import create_seed_dataset, create_text_dataset


def cmd_list(args):
    """List all available sprites."""
    print(f"\n{'='*60}")
    print(f"Available sprites ({len(SPRITES)} total)")
    print(f"{'='*60}")

    for cat in CATEGORIES:
        sprites = list_sprites(cat)
        if not sprites:
            continue
        print(f"\n[{cat.upper()}] ({len(sprites)} sprites)")
        for name in sprites:
            spec = SPRITES[name]
            print(f"  {name:<25} {spec['size']:>3}×{spec['size']:<3} — {spec.get('description', '')}")

    print(f"\nAvailable palettes: {', '.join(PALETTES.keys())}")


def cmd_preview(args):
    """Generate a sprite sheet preview."""
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    # Choose palette
    if args.palette in PALETTES:
        palette = PALETTES[args.palette]
    else:
        palette = PaletteGenerator.generate(
            base_hue=float(args.palette) if args.palette.isdigit() else 120,
            harmony="triadic",
            num_shades=4,
        )

    print(f"\nGenerating sprite sheet with palette: {palette.name}")

    # Filter sprites by category
    if args.category:
        sprite_names = list_sprites(args.category)
    else:
        sprite_names = list(SPRITES.keys())

    print(f"  Rendering {len(sprite_names)} sprites at {args.scale}× scale...")

    sheet = generate_sprite_sheet(
        sprite_names=sprite_names,
        palette=palette,
        display_scale=args.scale,
        cols=8,
        bg_color=(15, 15, 25),
        padding=3,
        dither_method=args.dither,
    )

    # Also generate per-category sheets
    for cat in CATEGORIES:
        cat_sprites = list_sprites(cat)
        if not cat_sprites:
            continue

        cat_sheet = generate_sprite_sheet(
            sprite_names=cat_sprites,
            palette=palette,
            display_scale=args.scale,
            cols=min(8, len(cat_sprites)),
            bg_color=(15, 15, 25),
            padding=3,
            dither_method=args.dither,
        )
        cat_path = output / f"sheet_{cat}.png"
        cat_sheet.save(cat_path)
        print(f"  Saved {cat} sheet -> {cat_path}")

    # Save full sheet
    full_path = output / "sheet_all.png"
    sheet.save(full_path)
    print(f"\nSaved full sprite sheet -> {full_path}")


def cmd_single(args):
    """Render a single sprite with multiple palette variations."""
    if args.sprite not in SPRITES:
        print(f"Error: Unknown sprite '{args.sprite}'")
        print(f"Available: {', '.join(SPRITES.keys())}")
        sys.exit(1)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    spec = SPRITES[args.sprite]
    renderer = SpriteRenderer()

    print(f"\nRendering '{args.sprite}' ({spec['size']}×{spec['size']}) with {args.n_variations} color variations")

    palettes = generate_palette_variations(spec["palette_type"], args.n_variations)

    dither_modes = ["none", "bayer2x2", "bayer4x4", "bayer8x8", "floyd_steinberg", "atkinson"]

    for i, palette in enumerate(palettes):
        for dither in (dither_modes if args.all_dithers else [args.dither]):
            img = renderer.render_to_pil(
                args.sprite, palette,
                display_scale=args.scale,
                dither_method=dither,
                dither_intensity=0.35,
            )
            fname = f"{args.sprite}_v{i:02d}_{dither}.png"
            img.save(output / fname)
            print(f"  -> {fname}")

    # Also render a comparison sheet
    from pixelgan.data.sprite_generator import generate_sprite_sheet
    all_palettes = generate_palette_variations(spec["palette_type"], min(16, args.n_variations))
    comp_imgs = []

    for palette in all_palettes[:16]:
        img = renderer.render_to_pil(args.sprite, palette, display_scale=args.scale)
        comp_imgs.append(img)

    # Stitch into grid
    from PIL import Image
    import numpy as np
    W, H = comp_imgs[0].size
    cols = 8
    rows = (len(comp_imgs) + cols - 1) // cols
    grid = Image.new("RGBA", (cols * W + cols * 2, rows * H + rows * 2), (15, 15, 25, 255))
    for i, img in enumerate(comp_imgs):
        r, c = divmod(i, cols)
        grid.paste(img, (c * (W + 2) + 1, r * (H + 2) + 1))

    comp_path = output / f"{args.sprite}_comparison.png"
    grid.save(comp_path)
    print(f"\nSaved comparison -> {comp_path}")


def cmd_dataset(args):
    """Generate a full training dataset as parquet files."""
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    size = args.size
    n_per = args.n_per_sprite
    mode = args.dataset_mode

    print(f"\nGenerating {mode} dataset:")
    print(f"  Image size:     {size}×{size}")
    print(f"  Variations/sprite: {n_per}")
    print(f"  Output:         {output}")

    if args.category:
        sprite_names = list_sprites(args.category)
    else:
        sprite_names = list(SPRITES.keys())

    print(f"  Sprites:        {len(sprite_names)}")

    dither_methods = args.dither.split(",")
    samples = generate_training_batch(
        sprite_names=sprite_names,
        n_per_sprite=n_per,
        target_size=size,
        dither_methods=dither_methods,
        base_seed=args.seed,
        include_augmentations=not args.no_augment,
    )

    print(f"  Total samples:  {len(samples)}")

    # Save based on dataset mode
    if mode == "seed":
        image_bytes = [s["image_bytes"] for s in samples]
        seeds = [s["seed"] for s in samples]
        path = create_seed_dataset(
            image_bytes, seeds,
            str(output / f"sprites_seed_{size}x{size}.parquet")
        )

    elif mode == "text":
        image_bytes = [s["image_bytes"] for s in samples]
        captions = [s["caption"] for s in samples]
        path = create_text_dataset(
            image_bytes, captions,
            str(output / f"sprites_text_{size}x{size}.parquet")
        )

    # Also save a preview sheet
    print("\nGenerating preview sheet...")
    from pixelgan.data.sprite_generator import SpriteRenderer, generate_sprite_sheet
    palette = get_sprite_palette("player_cyan", seed=42)
    preview_names = list(SPRITES.keys())[:min(32, len(SPRITES))]

    sheet = generate_sprite_sheet(
        preview_names, palette,
        display_scale=max(2, 64 // size),
        cols=8, bg_color=(15, 15, 25), padding=3,
    )
    preview_path = output / f"preview_{size}x{size}.png"
    sheet.save(preview_path)
    print(f"Preview saved -> {preview_path}")

    print(f"\nDataset generation complete!")
    print(f"  Samples: {len(samples)}")
    print(f"  Format: {mode} (2-column parquet)")
    print(f"  Output: {output}")


def main():
    parser = argparse.ArgumentParser(
        description="PixelGAN Sprite Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=["list", "preview", "single", "dataset"],
                        default="preview", help="Operation mode")
    parser.add_argument("--output", default="datasets/sprites",
                        help="Output directory")
    parser.add_argument("--sprite", default="galaga_bee",
                        help="Sprite name (for --mode single)")
    parser.add_argument("--category", default=None,
                        choices=CATEGORIES + [None],
                        help="Filter by category")
    parser.add_argument("--palette", default="sweetie16",
                        help="Palette name or base hue 0-360")
    parser.add_argument("--scale", type=int, default=8,
                        help="Display scale factor (nearest-neighbor)")
    parser.add_argument("--size", type=int, default=32,
                        choices=[8, 16, 32, 64, 128, 256],
                        help="Target image size for dataset")
    parser.add_argument("--n-per-sprite", type=int, default=8,
                        help="Color variations per sprite")
    parser.add_argument("--n-variations", type=int, default=12,
                        help="Variations for single-sprite mode")
    parser.add_argument("--dither", default="none",
                        help="Dither method (or comma-separated list for dataset)")
    parser.add_argument("--all-dithers", action="store_true",
                        help="Generate all dither variants (single mode)")
    parser.add_argument("--dataset-mode", choices=["seed", "text"],
                        default="seed",
                        help="Dataset format type")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable flip augmentations")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    if args.mode == "list":
        cmd_list(args)
    elif args.mode == "preview":
        cmd_preview(args)
    elif args.mode == "single":
        cmd_single(args)
    elif args.mode == "dataset":
        cmd_dataset(args)


if __name__ == "__main__":
    main()
