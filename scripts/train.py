#!/usr/bin/env python3
"""
PixelGAN Training Script.

Trains a pixel art GAN on a parquet dataset.

Usage:
    # Train on seed dataset (unconditional)
    python scripts/train.py --size 32 --dataset datasets/sprites/sprites_seed_32x32.parquet

    # Train text->image
    python scripts/train.py --size 32 --dataset-type text \
        --dataset datasets/sprites/sprites_text_32x32.parquet

    # Train image->image
    python scripts/train.py --size 32 --dataset-type image_pair \
        --dataset datasets/pairs.parquet

    # Resume from checkpoint
    python scripts/train.py --size 32 --dataset ... --resume runs/pixelgan/checkpoints/

    # Quick test run
    python scripts/train.py --size 8 --dataset datasets/sprites/sprites_seed_8x8.parquet \
        --steps 500 --log-every 50

Performance notes:
  - 8×8  model: ~200k params, <0.1ms/step on GPU after JIT warmup
  - 32×32 model: ~900k params, ~0.5ms/step on GPU
  - Compare StyleGAN3 32×32: ~15M params, ~50ms/step
"""

import sys
import argparse
import os
from pathlib import Path

# Force line-buffered stdout so logs appear immediately when redirected to a file
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PixelGAN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Architecture
    parser.add_argument("--size", type=int, default=32,
                        choices=[8, 16, 32, 64, 128, 256],
                        help="Image size (8=NES, 16=retro icon, 32=SNES, 64=N64, 128, 256)")
    parser.add_argument("--channels", type=int, default=3,
                        choices=[3, 4],
                        help="Image channels (3=RGB on neutral bg, 4=RGBA)")
    parser.add_argument("--output-mode", default="rgb",
                        choices=["rgb", "palette_indexed"],
                        help="Generator output mode. "
                             "'rgb' = standard [-1,1] RGB output. "
                             "'palette_indexed' = G outputs N palette logits per pixel; "
                             "a differentiable PaletteLookup maps them to RGB for D. "
                             "Requires an indexed-format dataset (run convert_to_indexed.py first).")
    parser.add_argument("--palette-colors", type=int, default=8,
                        help="Palette size for palette_indexed mode (default: 8). "
                             "Must match the value used in convert_to_indexed.py.")

    # Dataset
    parser.add_argument("--dataset", required=True,
                        help="Path to parquet dataset file or directory")
    parser.add_argument("--dataset-type", default="seed",
                        choices=["seed", "text", "image_pair"],
                        help="Dataset type (determines conditioning mode)")

    # Training
    parser.add_argument("--steps", type=int, default=None,
                        help="Training steps (None = use total_kimg from config)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size (None = use config default)")
    parser.add_argument("--g-lr", type=float, default=None,
                        help="Generator learning rate")
    parser.add_argument("--d-lr", type=float, default=None,
                        help="Discriminator learning rate")
    parser.add_argument("--r1-gamma", type=float, default=None,
                        help="R1 gradient penalty strength")
    parser.add_argument("--total-kimg", type=int, default=None,
                        help="Training duration in kimg")

    # Output
    parser.add_argument("--output", default="runs/pixelgan",
                        help="Output directory")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint directory to resume from")
    parser.add_argument("--no-clear", action="store_true",
                        help="Keep existing samples/checkpoints instead of wiping them on start")

    # Logging
    parser.add_argument("--log-every", type=int, default=100,
                        help="Print metrics every N steps")
    parser.add_argument("--sample-every", type=int, default=500,
                        help="Save samples every N steps")
    parser.add_argument("--checkpoint-every", type=int, default=2000,
                        help="Save checkpoint every N steps")

    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--jax-platform", default=None,
                        help="JAX platform override (cpu/gpu/tpu)")
    parser.add_argument("--gpu-mem-fraction", type=float, default=0.75,
                        help="Fraction of GPU VRAM JAX may use (0.0-1.0). "
                             "Lower this if your system lags or runs OOM. "
                             "Default 0.75 leaves 25%% headroom for desktop/display.")
    parser.add_argument("--no-prealloc", action="store_true",
                        help="Disable JAX GPU pre-allocation (grow memory on demand). "
                             "Slightly slower but never spikes VRAM at startup.")

    return parser.parse_args()


def main():
    args = parse_args()

    # ── GPU / system resource limits ─────────────────────────────────────
    # These MUST be set before JAX is imported — JAX reads them at init time.
    import os

    # Lower process niceness so the desktop stays responsive during training
    try:
        os.nice(10)  # 0=normal, 10=background, 19=idle
    except (AttributeError, PermissionError):
        pass  # Windows or permission denied — not critical

    # Cap VRAM usage: JAX pre-allocates all GPU memory by default, which
    # causes system-wide lag. 0.75 leaves ~25% for the display driver + desktop.
    mem_frac = args.gpu_mem_fraction
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(mem_frac))

    # Grow memory on demand instead of one giant upfront allocation spike.
    # This prevents the "everything lagged" freeze during JIT compilation.
    if args.no_prealloc:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    else:
        # Still set mem fraction even with prealloc enabled
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")

    # Limit XLA to 8 CPU threads for compilation — avoids starving the desktop
    # during the JIT warmup phase (which would otherwise use all cores).
    os.environ.setdefault("XLA_FLAGS",
        "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=8")

    # Configure JAX platform
    if args.jax_platform:
        os.environ["JAX_PLATFORMS"] = args.jax_platform

    # Import JAX/Flax (triggers CUDA init — env vars must be set before this)
    import jax
    print(f"\nJAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print(f"GPU mem cap:   {mem_frac*100:.0f}% of VRAM  (use --gpu-mem-fraction to adjust)")

    from pixelgan.utils.config import get_config
    from pixelgan.training.dataset import load_dataset
    from pixelgan.training.trainer import PixelGANTrainer

    # Build config with overrides
    overrides = {}
    if args.batch_size:
        overrides["batch_size"] = args.batch_size
    if args.g_lr:
        overrides["g_lr"] = args.g_lr
    if args.d_lr:
        overrides["d_lr"] = args.d_lr
    if args.r1_gamma:
        overrides["r1_gamma"] = args.r1_gamma
    if args.total_kimg:
        overrides["total_kimg"] = args.total_kimg
    overrides["seed"] = args.seed
    overrides["dataset_type"] = args.dataset_type
    overrides["image_channels"] = args.channels

    cfg = get_config(args.size, **overrides)
    cfg.training.output_dir = args.output

    # Option A: palette-indexed output mode
    cfg.arch.output_mode      = args.output_mode
    cfg.arch.n_palette_colors = args.palette_colors

    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    _ds_kwargs = {}
    if args.dataset_type == "seed":
        # Pass n_palette_slots so SeedDataset returns the palette column
        # when it detects an indexed-format parquet (Option B).
        _ds_kwargs["n_palette_slots"] = args.palette_colors
    dataset = load_dataset(
        path=args.dataset,
        dataset_type=args.dataset_type,
        image_size=args.size,
        image_channels=args.channels,
        split="train",
        **_ds_kwargs,
    )
    print(f"  Dataset size: {len(dataset)} samples")

    # Create trainer — clear previous run unless resuming or --no-clear
    clear = not args.resume and not args.no_clear
    trainer = PixelGANTrainer(cfg, output_dir=args.output, clear=clear)

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Print training summary
    print(f"\n{'='*60}")
    print(f"Training Summary")
    print(f"{'='*60}")
    print(f"  Model size:    {args.size}×{args.size} pixel art")
    print(f"  Output mode:   {args.output_mode}"
          + (f" ({args.palette_colors} colours)"
             if args.output_mode == "palette_indexed" else ""))
    print(f"  Dataset type:  {args.dataset_type}")
    print(f"  Batch size:    {cfg.training.batch_size}")
    print(f"  G LR:          {cfg.training.g_lr}")
    print(f"  D LR:          {cfg.training.d_lr}")
    print(f"  R1 gamma:      {cfg.training.r1_gamma}")
    print(f"  ADA target:    {cfg.training.ada_target}")

    steps = args.steps
    if steps is None:
        steps = (cfg.training.total_kimg * 1000) // cfg.training.batch_size

    print(f"  Total steps:   {steps:,}")
    print(f"  Total kimg:    {steps * cfg.training.batch_size / 1000:.0f}")
    print(f"{'='*60}\n")

    # Start training!
    trainer.fit(
        dataset=dataset,
        steps=steps,
        log_every=args.log_every,
        sample_every=args.sample_every,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()
