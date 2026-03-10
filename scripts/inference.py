#!/usr/bin/env python3
"""
PixelGAN Inference Script.

Generate pixel art from a trained PixelGAN model.

Modes:
  - seed: Generate images from random seeds (unconditional)
  - text: Generate images from text prompts
  - interpolate: Interpolate between two latent codes
  - grid: Generate a grid of samples

Usage:
    # Generate 16 random sprites
    python scripts/inference.py --checkpoint runs/pixelgan/checkpoints \
        --size 32 --n-samples 16

    # Generate from specific seeds
    python scripts/inference.py --checkpoint runs/... --mode seed \
        --seeds 42,1337,9999

    # Text-conditioned generation
    python scripts/inference.py --checkpoint runs/... --mode text \
        --prompts "galaga bee alien" "zelda link character"

    # Latent interpolation (morphing between sprites)
    python scripts/inference.py --checkpoint runs/... --mode interpolate \
        --seed-a 42 --seed-b 1337 --n-steps 8

    # Quality/diversity tradeoff via truncation
    python scripts/inference.py --checkpoint runs/... --truncation 0.5
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="PixelGAN Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to checkpoint directory or .pkl file")
    parser.add_argument("--size", type=int, default=32,
                        choices=[8, 32, 64, 128, 256])
    parser.add_argument("--channels", type=int, default=4, choices=[3, 4])
    parser.add_argument("--mode", default="seed",
                        choices=["seed", "text", "interpolate", "grid"])
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--seeds", default=None,
                        help="Comma-separated seeds (e.g. '42,1337')")
    parser.add_argument("--prompts", nargs="*",
                        help="Text prompts for text mode")
    parser.add_argument("--seed-a", type=int, default=42)
    parser.add_argument("--seed-b", type=int, default=1337)
    parser.add_argument("--n-steps", type=int, default=8,
                        help="Interpolation steps")
    parser.add_argument("--truncation", type=float, default=0.7,
                        help="Truncation psi (1.0=diverse, 0.5=quality)")
    parser.add_argument("--output", default="outputs",
                        help="Output directory")
    parser.add_argument("--display-scale", type=int, default=8,
                        help="Nearest-neighbor upscale for saved images")
    parser.add_argument("--dataset-type", default="seed",
                        choices=["seed", "text", "image_pair"])
    return parser.parse_args()


def load_model(args):
    """Load generator from checkpoint."""
    import jax
    import jax.numpy as jnp
    import pickle
    import numpy as np
    from pixelgan.utils.config import get_config
    from pixelgan.models.generator import make_generator
    from pixelgan.training.trainer import GANTrainState
    import optax

    # Resolve checkpoint path — supports both:
    #   step_XXXXXX/checkpoint.pkl  (directory-based, trainer default)
    #   step_XXXXXX.pkl             (flat file, checkpoint.py utility)
    path = Path(args.checkpoint)
    if path.is_dir():
        # Look for step_XXXXXX/checkpoint.pkl (trainer format)
        ckpts = sorted(path.glob("step_*/checkpoint.pkl"))
        if not ckpts:
            # Fallback: flat step_*.pkl files
            ckpts = sorted(path.glob("checkpoints/step_*.pkl"))
        if not ckpts:
            ckpts = sorted(path.glob("step_*.pkl"))
        if not ckpts:
            raise FileNotFoundError(
                f"No checkpoints found in {path}\n"
                f"Expected: {path}/step_XXXXXX/checkpoint.pkl"
            )
        ckpt_path = ckpts[-1]
    elif path.is_file():
        ckpt_path = path
    else:
        # Maybe it's a step directory itself
        ckpt_path = path / "checkpoint.pkl"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)

    # Auto-detect image size from checkpoint weight shapes
    try:
        synthesis_params = ckpt["g_ema_params"]["synthesis"]
        blocks = [k for k in synthesis_params.keys() if k.startswith("block_")]
        detected_size = 4 * (2 ** len(blocks))
        if detected_size != args.size:
            print(f"  Note: checkpoint is --size {detected_size}, overriding --size {args.size}")
            args.size = detected_size
    except (KeyError, TypeError):
        pass  # Can't detect, use args.size as-is

    cfg = get_config(
        args.size,
        image_channels=args.channels,
        dataset_type=args.dataset_type,
    )
    generator = make_generator(cfg)

    # We only need EMA params for inference
    ema_params = jax.tree_util.tree_map(jnp.array, ckpt["g_ema_params"])
    ema_vars = jax.tree_util.tree_map(jnp.array, ckpt.get("g_ema_vars", {}))
    step = ckpt.get("step", 0)
    kimg = ckpt.get("cur_kimg", 0)
    print(f"  Detected size: {args.size}×{args.size} | Step: {step:,} | kimg: {kimg:.1f}")

    return generator, ema_params, ema_vars, cfg


def save_image_grid(images_np, output_path, scale=4):
    """Save a grid of images."""
    from PIL import Image
    import numpy as np

    B, H, W, C = images_np.shape
    n_cols = min(8, B)
    n_rows = (B + n_cols - 1) // n_cols

    grid_h = n_rows * H * scale + n_rows * 2
    grid_w = n_cols * W * scale + n_cols * 2
    mode = "RGBA" if C == 4 else "RGB"
    grid = Image.new(mode, (grid_w, grid_h), (15, 15, 25, 255) if C == 4 else (15, 15, 25))

    for i, img_arr in enumerate(images_np):
        r, c = divmod(i, n_cols)
        img_big = np.repeat(np.repeat(img_arr, scale, axis=0), scale, axis=1)
        x = c * (W * scale + 2) + 1
        y = r * (H * scale + 2) + 1
        grid.paste(Image.fromarray(img_big, mode=mode), (x, y))

    grid.save(output_path)
    print(f"Saved -> {output_path}")


def main():
    args = parse_args()

    import jax
    import jax.numpy as jnp
    import numpy as np

    print(f"JAX: {jax.default_backend()} | Devices: {jax.devices()}")

    generator, ema_params, ema_vars, cfg = load_model(args)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    z_dim = cfg.arch.z_dim

    if args.mode == "seed":
        # Parse seeds or generate random ones
        if args.seeds:
            seed_list = [int(s) for s in args.seeds.split(",")]
        else:
            seed_list = list(range(args.n_samples))

        print(f"\nGenerating {len(seed_list)} images from seeds: {seed_list[:8]}...")

        z_list = []
        for seed in seed_list:
            z = jax.random.normal(jax.random.PRNGKey(seed), (z_dim,))
            z_list.append(np.array(z))

        z = jnp.array(np.stack(z_list))  # [N, z_dim]

        images = generator.apply(
            {"params": ema_params, "ema": ema_vars},
            z, None,
            truncation_psi=args.truncation,
            train=False,
            mutable=False,
        )

        images_np = ((np.array(images) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        save_image_grid(images_np, output / "generated_seeds.png", args.display_scale)

        # Also save individual images
        for i, (seed, img) in enumerate(zip(seed_list, images_np)):
            import io
            from PIL import Image
            img_big = np.repeat(np.repeat(img, args.display_scale, axis=0),
                                args.display_scale, axis=1)
            mode = "RGBA" if img.shape[2] == 4 else "RGB"
            Image.fromarray(img_big, mode=mode).save(output / f"seed_{seed:06d}.png")

    elif args.mode == "text":
        from pixelgan.training.dataset import tokenize_text

        prompts = args.prompts or [
            "galaga bee alien purple",
            "zelda link green character",
            "pacman ghost red blinky",
            "gold star item pixel art",
        ]

        print(f"\nGenerating {len(prompts)} images from text prompts...")

        tokens_list = [tokenize_text(p, cfg.arch.text_max_length) for p in prompts]
        tokens = jnp.array(np.stack(tokens_list))  # [N, seq_len]

        z = jax.random.normal(jax.random.PRNGKey(42), (len(prompts), z_dim))

        images = generator.apply(
            {"params": ema_params, "ema": ema_vars},
            z, tokens,
            truncation_psi=args.truncation,
            train=False,
            mutable=False,
        )

        images_np = ((np.array(images) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        save_image_grid(images_np, output / "generated_text.png", args.display_scale)

        for prompt, img in zip(prompts, images_np):
            fname = prompt.replace(" ", "_")[:30] + ".png"
            img_big = np.repeat(np.repeat(img, args.display_scale, axis=0),
                                args.display_scale, axis=1)
            mode = "RGBA" if img.shape[2] == 4 else "RGB"
            from PIL import Image
            Image.fromarray(img_big, mode=mode).save(output / fname)

    elif args.mode == "interpolate":
        print(f"\nInterpolating from seed {args.seed_a} -> {args.seed_b} ({args.n_steps} steps)...")

        za = jax.random.normal(jax.random.PRNGKey(args.seed_a), (z_dim,))
        zb = jax.random.normal(jax.random.PRNGKey(args.seed_b), (z_dim,))

        # Spherical interpolation (slerp) for latent space
        def slerp(a, b, t):
            a_norm = a / (jnp.linalg.norm(a) + 1e-8)
            b_norm = b / (jnp.linalg.norm(b) + 1e-8)
            dot = jnp.clip(jnp.dot(a_norm, b_norm), -1.0, 1.0)
            omega = jnp.arccos(dot)
            # Handle near-parallel case
            sin_omega = jnp.sin(omega)
            safe = jnp.where(
                sin_omega < 1e-6,
                a * (1 - t) + b * t,
                jnp.sin((1 - t) * omega) / sin_omega * a +
                jnp.sin(t * omega) / sin_omega * b,
            )
            return safe

        ts = jnp.linspace(0, 1, args.n_steps)
        z_interp = jnp.stack([slerp(za, zb, t) for t in ts])  # [N, z_dim]

        images = generator.apply(
            {"params": ema_params, "ema": ema_vars},
            z_interp, None,
            truncation_psi=args.truncation,
            train=False,
            mutable=False,
        )

        images_np = ((np.array(images) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        save_image_grid(images_np, output / "interpolation.png", args.display_scale)

        # Also save as animated GIF
        try:
            from PIL import Image
            frames = []
            for img in images_np:
                img_big = np.repeat(np.repeat(img, args.display_scale, axis=0),
                                    args.display_scale, axis=1)
                mode = "RGBA" if img.shape[2] == 4 else "RGB"
                frames.append(Image.fromarray(img_big, mode=mode).convert("RGBA"))

            gif_path = output / "interpolation.gif"
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:] + frames[::-1][1:],  # ping-pong
                duration=150,
                loop=0,
            )
            print(f"Saved GIF -> {gif_path}")
        except Exception as e:
            print(f"  (GIF creation failed: {e})")

    elif args.mode == "grid":
        n = args.n_samples
        print(f"\nGenerating {n}×{n} sample grid...")

        z = jax.random.normal(jax.random.PRNGKey(args.seed_a), (n, z_dim))
        images = generator.apply(
            {"params": ema_params, "ema": ema_vars},
            z, None,
            truncation_psi=args.truncation,
            train=False,
            mutable=False,
        )

        images_np = ((np.array(images) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        save_image_grid(images_np, output / "grid.png", args.display_scale)

    print(f"\nDone! Output saved to: {output}")


if __name__ == "__main__":
    main()
