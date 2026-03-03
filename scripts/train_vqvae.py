#!/usr/bin/env python3
"""
train_vqvae.py — Option C Stage 1: Train the VQ-VAE on your sprite dataset.

Trains a Vector-Quantized Variational Autoencoder to compress pixel art
sprites from 64×64×3 into a compact 8×8 discrete latent space.

After training, the saved decoder is used directly by the GAN (Stage 2):
the generator produces 8×8 latent codes → frozen decoder → 64×64 image.

This gives the GAN 64× fewer values to model, making training faster and
the generator architecture far smaller.

Usage:
    # Basic: train for 10k steps on tree sprites
    python scripts/train_vqvae.py \
        --dataset datasets/sprites/trees.parquet \
        --size 64 \
        --steps 10000 \
        --output runs/vqvae_trees

    # With custom codebook:
    python scripts/train_vqvae.py \
        --dataset datasets/sprites/trees.parquet \
        --size 64 \
        --codebook-size 512 \
        --latent-dim 128 \
        --steps 20000

    # Resume from checkpoint:
    python scripts/train_vqvae.py \
        --dataset datasets/sprites/trees.parquet \
        --size 64 \
        --resume runs/vqvae_trees/checkpoint/latest \
        --steps 5000

GPU safety flags are applied automatically (same as train.py):
  XLA_PYTHON_CLIENT_PREALLOCATE=false
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
"""

import argparse
import os
import sys


# ---------------------------------------------------------------------------
# GPU safety — MUST be before any JAX import
# ---------------------------------------------------------------------------

def _apply_gpu_safety(prealloc: bool, mem_fraction: float, n_cpu_threads: int) -> None:
    if not prealloc:
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault(
        "XLA_PYTHON_CLIENT_MEM_FRACTION", str(mem_fraction)
    )
    os.environ.setdefault(
        "XLA_FLAGS",
        f"--xla_cpu_multi_thread_eigen=true "
        f"intra_op_parallelism_threads={n_cpu_threads}",
    )
    os.nice(10)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 1 VQ-VAE training for PixelGAN (Option C).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Data ---
    p.add_argument("--dataset", required=True,
                   help="Path to .parquet dataset (seed+image columns).")
    p.add_argument("--size", type=int, default=64, choices=[8, 16, 32, 64, 128, 256],
                   help="Image size (default: 64).")

    # --- VQ-VAE architecture ---
    p.add_argument("--codebook-size", type=int, default=256,
                   help="Number of discrete codes K (default: 256).")
    p.add_argument("--latent-dim", type=int, default=64,
                   help="Code vector dimension D (default: 64).")
    p.add_argument("--base-channels", type=int, default=64,
                   help="Encoder/decoder base channel count (default: 64).")
    p.add_argument("--n-res-blocks", type=int, default=2,
                   help="Residual blocks per scale (default: 2).")
    p.add_argument("--commitment-beta", type=float, default=0.25,
                   help="VQ commitment loss weight (default: 0.25).")

    # --- Training ---
    p.add_argument("--steps", type=int, default=10_000,
                   help="Number of gradient steps (default: 10000).")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Training batch size (default: 16).")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Adam learning rate (default: 1e-3).")
    p.add_argument("--lambda-recon", type=float, default=1.0,
                   help="Reconstruction loss weight (default: 1.0).")
    p.add_argument("--lambda-vq", type=float, default=1.0,
                   help="VQ commitment+codebook loss weight (default: 1.0).")
    p.add_argument("--snapshot-steps", type=int, default=1000,
                   help="Save checkpoint + reconstruction grid every N steps.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed.")

    # --- Output ---
    p.add_argument("--output", "-o", default="runs/vqvae",
                   help="Output directory (default: runs/vqvae).")
    p.add_argument("--resume", default=None,
                   help="Path to checkpoint directory to resume from.")

    # --- GPU safety ---
    p.add_argument("--gpu-mem-fraction", type=float, default=0.75,
                   help="Fraction of GPU VRAM to pre-allocate (default: 0.75).")
    p.add_argument("--prealloc", action="store_true",
                   help="Enable JAX VRAM pre-allocation (default: off).")
    p.add_argument("--cpu-threads", type=int, default=8,
                   help="Max XLA CPU threads (default: 8).")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Apply GPU safety before importing JAX
    _apply_gpu_safety(
        prealloc=args.prealloc,
        mem_fraction=args.gpu_mem_fraction,
        n_cpu_threads=args.cpu_threads,
    )

    # Now safe to import JAX
    import jax

    # Add project root to path so we can import pixelgan
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))

    from pixelgan.utils.config import get_config, VQVAEConfig
    from pixelgan.training.dataset import load_dataset
    from pixelgan.training.vqvae_trainer import VQVAETrainer

    # ------------------------------------------------------------------
    # Build config
    # ------------------------------------------------------------------
    print(f"\n[train_vqvae] Building config for {args.size}×{args.size} sprites")

    config = get_config(args.size)
    config.training.seed = args.seed

    config.vqvae = VQVAEConfig(
        codebook_size    = args.codebook_size,
        latent_dim       = args.latent_dim,
        base_channels    = args.base_channels,
        n_res_blocks     = args.n_res_blocks,
        commitment_beta  = args.commitment_beta,
        lr               = args.lr,
        batch_size       = args.batch_size,
        total_steps      = args.steps,
        snapshot_steps   = args.snapshot_steps,
        lambda_recon     = args.lambda_recon,
        lambda_vq        = args.lambda_vq,
        checkpoint_path  = f"{args.output}/checkpoint",
    )

    print(f"  VQ-VAE config:")
    print(f"    codebook_size   = {args.codebook_size}")
    print(f"    latent_dim      = {args.latent_dim}")
    print(f"    base_channels   = {args.base_channels}")
    print(f"    latent_grid     = {args.size // 8}×{args.size // 8} "
          f"({(args.size // 8) ** 2} positions vs {args.size ** 2} pixels)")
    print(f"    commitment_beta = {args.commitment_beta}")
    print(f"  Training:")
    print(f"    steps       = {args.steps:,}")
    print(f"    batch_size  = {args.batch_size}")
    print(f"    lr          = {args.lr}")
    print(f"    output      = {args.output}")

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    print(f"\n[train_vqvae] Loading dataset: {args.dataset}")
    dataset = load_dataset(
        args.dataset,
        dataset_type="seed",
        image_size=args.size,
        image_channels=3,   # VQ-VAE trains on RGB (alpha composited by decoder)
        split="train",
    )
    print(f"  {len(dataset):,} training samples")

    # ------------------------------------------------------------------
    # Build trainer and run
    # ------------------------------------------------------------------
    print(f"\n[train_vqvae] Initialising VQVAETrainer...")
    trainer = VQVAETrainer(config, output_dir=args.output)

    if args.resume:
        print(f"[train_vqvae] Resuming from {args.resume}")
        # Need state initialised before loading checkpoint
        sample    = dataset.get_batch(1)
        import jax.numpy as jnp
        trainer.state = trainer._build_state(jnp.array(sample["image"][:1]))
        trainer.load_checkpoint(args.resume)

    print(f"\n[train_vqvae] Starting training on {jax.device_count()} device(s)")

    metrics = trainer.fit(dataset, steps=args.steps)

    print("\n[train_vqvae] Final metrics:")
    for k, v in metrics.items():
        print(f"  {k:20s} = {v:.6f}")

    print(f"\n[train_vqvae] Checkpoint saved to: {args.output}/checkpoint/latest")
    print(f"[train_vqvae] Use --vqvae-path {args.output}/checkpoint/latest in train.py "
          f"to enable Stage 2 GAN training in the latent space.")


if __name__ == "__main__":
    main()
