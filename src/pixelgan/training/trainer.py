"""
PixelGAN Training Loop.

Full GAN training with:
  - Non-saturating logistic loss (StyleGAN2 style)
  - R1 gradient penalty for discriminator
  - Path length regularization for generator
  - Generator EMA (exponential moving average)
  - ADA (Adaptive Discriminator Augmentation) for small datasets
  - Three training modes: seed2img, text2img, img2img
  - JIT-compiled training step for maximum speed

JAX training state management:
  - Two separate TrainState objects (Generator, Discriminator)
  - Pure functional updates (no mutable state during training)
  - Explicit PRNG key threading
  - orbax-checkpoint for saving/loading

Performance design:
  - Full JIT on train_step_g and train_step_d
  - scan over gradient accumulation steps
  - No Python loops in the hot path
"""

from __future__ import annotations

import os
import time
from functools import partial
from pathlib import Path
from typing import Optional, Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.training import train_state
from flax import traverse_util

from ..models.generator import PixelArtGenerator, make_generator
from ..models.discriminator import PixelArtDiscriminator, make_discriminator
from ..training.losses import (
    generator_loss, discriminator_loss,
    compute_g_loss, compute_d_loss,
)
from ..utils.config import PixelGANConfig, TrainingConfig


# ---------------------------------------------------------------------------
# Custom TrainState with EMA support
# ---------------------------------------------------------------------------

class GANTrainState(train_state.TrainState):
    """Extended train state with EMA parameters and extra state."""
    ema_params: dict  # Generator EMA parameters
    ema_vars: dict   # Mutable 'ema' collection (e.g. w_avg for truncation trick)


# ---------------------------------------------------------------------------
# ADA (Adaptive Discriminator Augmentation) — simple version
# ---------------------------------------------------------------------------

def ada_augment(
    images: jnp.ndarray,    # [B, H, W, C]
    p: float,               # augmentation probability [0, 1]
    rng: jax.random.KeyArray,
) -> jnp.ndarray:
    """
    Apply random augmentations to images with probability p.

    For pixel art we use only:
      - Horizontal flip
      - Vertical flip (less common, but helps)
      - 90° rotations
      - Color jitter (slight brightness/contrast)

    We intentionally AVOID:
      - Blurring (destroys pixel art crispness)
      - Translation (affects pixel-perfect alignment)
      - Scaling (would need resize)

    Args:
        images: [B, H, W, C] float32 in [-1, 1]
        p: Probability of applying each augmentation
        rng: PRNG key

    Returns:
        Augmented images [B, H, W, C]
    """
    rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)
    B = images.shape[0]

    # Horizontal flip with prob p
    do_hflip = jax.random.uniform(rng1, (B,)) < p
    images = jnp.where(
        do_hflip[:, None, None, None],
        images[:, :, ::-1, :],
        images
    )

    # 90° rotation with prob p/2 (rarer for typical sprites)
    do_rot = jax.random.uniform(rng2, (B,)) < (p * 0.5)
    rotated = jnp.rot90(images, k=1, axes=(1, 2))
    images = jnp.where(do_rot[:, None, None, None], rotated, images)

    # Slight brightness jitter with prob p
    do_bright = jax.random.uniform(rng3, (B,)) < p
    bright_delta = jax.random.uniform(rng4, (B, 1, 1, 1), minval=-0.1, maxval=0.1)
    images = jnp.where(
        do_bright[:, None, None, None],
        jnp.clip(images + bright_delta, -1.0, 1.0),
        images
    )

    return images


# ---------------------------------------------------------------------------
# Training step functions (pure JAX, JIT-compiled)
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnums=(0, 1, 9))
def train_step_d(
    generator: PixelArtGenerator,
    discriminator: PixelArtDiscriminator,
    g_state: train_state.TrainState,
    d_state: train_state.TrainState,
    real_images: jnp.ndarray,        # [B, H, W, C]
    z: jnp.ndarray,                  # [B, z_dim] latent codes
    condition: Optional[jnp.ndarray],  # Conditioning (type depends on mode)
    ada_p: float,                     # ADA augmentation probability
    r1_gamma: float,                  # R1 penalty weight
    apply_r1: bool,                   # Whether to apply R1 this step
    rng: jax.random.KeyArray,
) -> tuple[train_state.TrainState, dict]:
    """
    Discriminator training step.

    Returns updated d_state and loss metrics.
    """
    rng, noise_rng, aug_rng_real, aug_rng_fake = jax.random.split(rng, 4)

    # Generate fake images (no gradient through generator)
    fake_images = generator.apply(
        {"params": g_state.params, "ema": g_state.ema_vars},
        z,
        condition,
        train=False,
        rng=noise_rng,
        mutable=False,
    )
    fake_images = jax.lax.stop_gradient(fake_images)

    # Apply ADA augmentation
    real_aug = ada_augment(real_images, ada_p, aug_rng_real)
    fake_aug = ada_augment(fake_images, ada_p, aug_rng_fake)

    def d_loss_fn(d_params):
        real_logits = discriminator.apply({"params": d_params}, real_aug)
        fake_logits = discriminator.apply({"params": d_params}, fake_aug)

        losses = compute_d_loss(real_logits, fake_logits)
        total = losses["total"]

        # R1 gradient penalty (applied lazily every r1_interval steps)
        r1_loss = jnp.zeros(())
        if apply_r1:
            def d_logit_sum(images):
                return jnp.sum(
                    discriminator.apply({"params": d_params}, images)
                )
            r1_grads = jax.grad(d_logit_sum)(real_aug)
            r1_penalty = jnp.mean(jnp.sum(r1_grads ** 2, axis=(1, 2, 3)))
            r1_loss = (r1_gamma / 2.0) * r1_penalty
            total = total + r1_loss

        return total, {**losses, "r1": r1_loss}

    (loss, metrics), grads = jax.value_and_grad(d_loss_fn, has_aux=True)(
        d_state.params
    )

    d_state = d_state.apply_gradients(grads=grads)

    return d_state, metrics


@partial(jax.jit, static_argnums=(0, 1))
def train_step_g(
    generator: PixelArtGenerator,
    discriminator: PixelArtDiscriminator,
    g_state: GANTrainState,
    d_state: train_state.TrainState,
    z: jnp.ndarray,                   # [B, z_dim]
    condition: Optional[jnp.ndarray],
    ada_p: float,
    ema_beta: float,                   # EMA decay for generator
    rng: jax.random.KeyArray,
) -> tuple[GANTrainState, dict]:
    """
    Generator training step.

    Returns updated g_state (with EMA) and loss metrics.
    """
    rng, noise_rng, aug_rng = jax.random.split(rng, 3)

    g_ema_vars = g_state.ema_vars

    def g_loss_fn(g_params):
        fake_images = generator.apply(
            {"params": g_params, "ema": g_ema_vars},
            z,
            condition,
            train=True,
            rng=noise_rng,
            mutable=False,
        )

        # Augment fake images before discriminating
        fake_aug = ada_augment(fake_images, ada_p, aug_rng)

        fake_logits = discriminator.apply(
            {"params": d_state.params}, fake_aug
        )

        losses = compute_g_loss(fake_logits, fake_images)
        return losses["total"], {**losses, "fake_images": fake_images}

    (loss, metrics), grads = jax.value_and_grad(g_loss_fn, has_aux=True)(
        g_state.params
    )

    g_state = g_state.apply_gradients(grads=grads)

    # Update EMA parameters
    new_ema = jax.tree_util.tree_map(
        lambda ema, p: ema * ema_beta + p * (1 - ema_beta),
        g_state.ema_params,
        g_state.params,
    )
    g_state = g_state.replace(ema_params=new_ema)

    return g_state, metrics


# ---------------------------------------------------------------------------
# Sample generation (uses EMA generator)
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnums=(0,))
def generate_samples(
    generator: PixelArtGenerator,
    g_state: GANTrainState,
    z: jnp.ndarray,
    condition: Optional[jnp.ndarray] = None,
    rng: Optional[jax.random.KeyArray] = None,
) -> jnp.ndarray:
    """Generate images using EMA generator parameters."""
    return generator.apply(
        {"params": g_state.ema_params, "ema": g_state.ema_vars},
        z,
        condition,
        train=False,
        rng=rng,
        truncation_psi=0.7,
        mutable=False,
    )


# ---------------------------------------------------------------------------
# Main Trainer class
# ---------------------------------------------------------------------------

class PixelGANTrainer:
    """
    High-level trainer for PixelGAN.

    Manages:
      - Model initialization
      - Training loop (G and D steps)
      - EMA tracking
      - ADA augmentation probability
      - Checkpointing
      - Sample generation and logging

    Usage:
        cfg = get_config(32)
        trainer = PixelGANTrainer(cfg)
        trainer.fit(dataset, epochs=100)
    """

    def __init__(
        self,
        config: PixelGANConfig,
        output_dir: Optional[str] = None,
        clear: bool = True,
    ):
        self.config = config
        self.arch = config.arch
        self.tcfg = config.training
        self.output_dir = Path(output_dir or config.training.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if clear:
            self._clear_output_dir()

        # Build models
        self.generator = make_generator(config)
        self.discriminator = make_discriminator(config)

        # Initialize JAX random state
        self.rng = jax.random.PRNGKey(config.training.seed)

        # Training counters
        self.cur_kimg = 0
        self.cur_tick = 0
        self.ada_p = 0.0  # ADA augmentation probability

        # Initialize training states
        self._init_states()

        print(f"PixelGAN initialized:")
        print(f"  Image size:     {self.arch.image_size}×{self.arch.image_size}")
        print(f"  Dataset type:   {self.tcfg.dataset_type}")
        print(f"  Output:         {self.output_dir}")
        print(f"  G params: {self._count_params(self.g_state.params):,}")
        print(f"  D params: {self._count_params(self.d_state.params):,}")

    def _clear_output_dir(self):
        """Remove previous samples and checkpoints from output_dir."""
        import shutil
        # Delete old sample PNGs
        for f in self.output_dir.glob("samples_*.png"):
            f.unlink()
        # Delete checkpoints subtree
        ckpt_dir = self.output_dir / "checkpoints"
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)

    def _count_params(self, params: dict) -> int:
        """Count total parameters in a pytree."""
        return sum(x.size for x in jax.tree_util.tree_leaves(params))

    def _init_states(self):
        """Initialize generator and discriminator training states."""
        self.rng, g_rng, d_rng, z_rng = jax.random.split(self.rng, 4)

        # Dummy inputs for initialization
        B = self.tcfg.batch_size
        H = W = self.arch.image_size
        C = self.arch.image_channels

        dummy_z = jnp.zeros((B, self.arch.z_dim))
        dummy_img = jnp.zeros((B, H, W, C))

        # Initialize generator (pass rng so noise_scale params are created)
        g_vars = self.generator.init(
            {"params": g_rng},
            dummy_z,
            None,
            train=True,
            rng=jax.random.PRNGKey(0),
        )

        # Initialize discriminator
        d_vars = self.discriminator.init(
            {"params": d_rng},
            dummy_img,
        )

        # Build optimizers
        g_optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(self.tcfg.g_lr, self.tcfg.g_beta1, self.tcfg.g_beta2),
        )
        d_optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(self.tcfg.d_lr, self.tcfg.d_beta1, self.tcfg.d_beta2),
        )

        g_params = g_vars["params"]

        g_ema_vars = g_vars.get("ema", {})

        self.g_state = GANTrainState.create(
            apply_fn=self.generator.apply,
            params=g_params,
            tx=g_optimizer,
            ema_params=g_params,  # Start EMA = params
            ema_vars=g_ema_vars,
        )

        self.d_state = train_state.TrainState.create(
            apply_fn=self.discriminator.apply,
            params=d_vars["params"],
            tx=d_optimizer,
        )

    def _compute_ema_beta(self) -> float:
        """EMA decay factor, with rampup at start of training."""
        B = self.tcfg.batch_size
        ema_nimg = self.tcfg.ema_kimg * 1000
        if self.tcfg.ema_rampup is not None:
            rampup_nimg = self.cur_kimg * 1000 * self.tcfg.ema_rampup
            ema_nimg = min(ema_nimg, rampup_nimg)
        ema_beta = 0.5 ** (B / max(ema_nimg, 1e-8))
        return float(ema_beta)

    def _update_ada(self, real_sign: float, batch_size: int):
        """
        Update ADA augmentation probability based on discriminator sign.

        Target: ada_target fraction of real images scored as real.
        If D is too good (sign > target), increase augmentation.
        If D is struggling (sign < target), decrease augmentation.
        """
        target = self.tcfg.ada_target
        ada_kimg = self.tcfg.ada_kimg
        ada_interval = self.tcfg.ada_interval

        adjustment = (
            np.sign(real_sign - target)
            * (batch_size * ada_interval)
            / (ada_kimg * 1000)
        )
        self.ada_p = float(np.clip(self.ada_p + adjustment, 0.0, 1.0))

    def _save_samples(self, step: int, n_samples: int = 16):
        """Generate and save sample images as an upscaled RGB grid."""
        try:
            from PIL import Image
            import numpy as np_

            self.rng, z_rng = jax.random.split(self.rng)
            z = jax.random.normal(z_rng, (n_samples, self.arch.z_dim))

            self.rng, gen_rng = jax.random.split(self.rng)
            images = generate_samples(self.generator, self.g_state, z, None, gen_rng)

            # Denormalize [-1,1] -> [0,255]
            images_np = np_.array(images)
            images_np = ((images_np + 1.0) * 127.5).clip(0, 255).astype(np_.uint8)

            n_cols = min(8, n_samples)
            n_rows = (n_samples + n_cols - 1) // n_cols
            H = W = self.arch.image_size
            C = self.arch.image_channels
            scale = max(1, 128 // H)  # Upscale tiny sprites for visibility

            grid_h = n_rows * H * scale
            grid_w = n_cols * W * scale

            if C == 4:
                # Composite RGBA output onto a checkerboard so the alpha is visible
                cs = max(scale, 8)
                checker = np_.indices((grid_h, grid_w)).sum(axis=0) // cs % 2
                bg = np_.where(checker[:, :, None] == 0, 204, 153).astype(np_.uint8)
                bg = np_.repeat(bg, 3, axis=2)  # grey tones -> RGB
                grid = bg.copy()
                for i, img in enumerate(images_np):
                    r, c = divmod(i, n_cols)
                    img_big = np_.repeat(np_.repeat(img, scale, axis=0), scale, axis=1)
                    y, x = r * H * scale, c * W * scale
                    rgb = img_big[:, :, :3].astype(np_.float32)
                    a   = img_big[:, :, 3:4].astype(np_.float32) / 255.0
                    bg_tile = bg[y:y+H*scale, x:x+W*scale].astype(np_.float32)
                    blended = (rgb * a + bg_tile * (1.0 - a)).clip(0, 255).astype(np_.uint8)
                    grid[y:y+H*scale, x:x+W*scale] = blended
                mode = "RGB"
            else:
                # RGB output: lay directly onto a dark background grid
                grid = np_.full((grid_h, grid_w, 3), 28, dtype=np_.uint8)
                for i, img in enumerate(images_np):
                    r, c = divmod(i, n_cols)
                    img_big = np_.repeat(np_.repeat(img, scale, axis=0), scale, axis=1)
                    y, x = r * H * scale, c * W * scale
                    grid[y:y+H*scale, x:x+W*scale] = img_big
                mode = "RGB"

            out_path = self.output_dir / f"samples_{step:06d}.png"
            Image.fromarray(grid, mode=mode).save(out_path)
            print(f"  Saved samples -> {out_path}")
        except Exception as e:
            print(f"  Warning: Could not save samples: {e}")

            out_path = self.output_dir / f"samples_{step:06d}.png"
            Image.fromarray(grid, mode=mode).save(out_path)
            print(f"  Saved samples -> {out_path}")
        except Exception as e:
            print(f"  Warning: Could not save samples: {e}")

    def _save_checkpoint(self, step: int):
        """Save model checkpoint."""
        try:
            import orbax.checkpoint as ocp

            ckpt = {
                "g_params": self.g_state.params,
                "g_ema_params": self.g_state.ema_params,
                "g_ema_vars": self.g_state.ema_vars,
                "d_params": self.d_state.params,
                "step": step,
                "cur_kimg": self.cur_kimg,
                "ada_p": self.ada_p,
            }
            ckpt_dir = self.output_dir / "checkpoints" / f"step_{step:06d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            # Simple numpy save as fallback
            import pickle
            with open(ckpt_dir / "checkpoint.pkl", "wb") as f:
                pickle.dump(jax.tree_util.tree_map(np.array, ckpt), f)

            print(f"  Saved checkpoint -> {ckpt_dir}")
        except Exception as e:
            print(f"  Warning: Could not save checkpoint: {e}")

    def load_checkpoint(self, path: str):
        """Load a checkpoint from disk."""
        import pickle
        ckpt_path = Path(path) / "checkpoint.pkl"
        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)

        self.g_state = self.g_state.replace(
            params=jax.tree_util.tree_map(jnp.array, ckpt["g_params"]),
            ema_params=jax.tree_util.tree_map(jnp.array, ckpt["g_ema_params"]),
        )
        self.d_state = self.d_state.replace(
            params=jax.tree_util.tree_map(jnp.array, ckpt["d_params"]),
        )
        self.cur_kimg = float(ckpt.get("cur_kimg", 0))
        self.ada_p = float(ckpt.get("ada_p", 0.0))
        print(f"Loaded checkpoint from {path} (kimg={self.cur_kimg:.1f})")

    def train_on_batch(
        self,
        batch: dict,
        step: int,
    ) -> dict:
        """
        Run one full training step (D update + G update).

        Args:
            batch: Dict from dataset loader
            step: Current training step

        Returns:
            metrics dict
        """
        B = self.tcfg.batch_size
        tcfg = self.tcfg

        # Extract real images and conditioning from batch
        if tcfg.dataset_type == "seed":
            real_images = jnp.array(batch["image"])
            condition = None
        elif tcfg.dataset_type == "text":
            real_images = jnp.array(batch["image"])
            condition = jnp.array(batch["tokens"])
        elif tcfg.dataset_type == "image_pair":
            real_images = jnp.array(batch["target"])
            condition = jnp.array(batch["source"])
        else:
            raise ValueError(f"Unknown dataset_type: {tcfg.dataset_type!r}")

        # Sample random latent codes
        self.rng, z_rng, d_rng, g_rng = jax.random.split(self.rng, 4)
        z = jax.random.normal(z_rng, (B, self.arch.z_dim))

        apply_r1 = (step % tcfg.r1_interval == 0)

        # ── Discriminator update ──────────────────────────────────────────
        # Scale r1_gamma by the interval: lazy reg applies (gamma*interval/2)*penalty
        # once every N steps so the *expected* penalty per step equals (gamma/2)*penalty.
        # Without this scaling, R1 is 16× weaker than intended and D's logits explode.
        r1_gamma_scaled = tcfg.r1_gamma * tcfg.r1_interval
        self.d_state, d_metrics = train_step_d(
            self.generator, self.discriminator,
            self.g_state, self.d_state,
            real_images, z, condition,
            self.ada_p, r1_gamma_scaled, apply_r1,
            d_rng,
        )

        # ── Generator update ──────────────────────────────────────────────
        self.rng, z_rng2 = jax.random.split(self.rng)
        z2 = jax.random.normal(z_rng2, (B, self.arch.z_dim))
        ema_beta = self._compute_ema_beta()

        self.g_state, g_metrics = train_step_g(
            self.generator, self.discriminator,
            self.g_state, self.d_state,
            z2, condition,
            self.ada_p, ema_beta,
            g_rng,
        )

        # Update ADA every ada_interval steps
        if step % tcfg.ada_interval == 0:
            real_sign = float(d_metrics.get("real_sign", 0.0))
            self._update_ada(real_sign, B)

        return {
            "g_loss": float(g_metrics["total"]),
            "d_loss": float(d_metrics["total"]),
            "d_real": float(d_metrics["real"]),
            "d_fake": float(d_metrics["fake"]),
            "r1": float(d_metrics.get("r1", 0.0)),
            "ada_p": self.ada_p,
            "ema_beta": ema_beta,
        }

    # ------------------------------------------------------------------
    # Rich terminal UI helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_bar_progress(label: str, color: str, width: int = 36):
        """Create a single-task Progress bar used as a metric gauge."""
        from rich.progress import Progress, BarColumn, TextColumn
        return Progress(
            TextColumn(f"[{color}]{label}[/{color}]"),
            BarColumn(bar_width=width, complete_style=color, finished_style=color),
            TextColumn("{task.fields[val]}", style="bold"),
            expand=False,
        )

    # ------------------------------------------------------------------

    def fit(
        self,
        dataset,
        steps: Optional[int] = None,
        log_every: int = 100,
        sample_every: int = 500,
        checkpoint_every: int = 2000,
    ):
        """
        Full training loop.  Shows a rich multi-bar terminal UI when
        stdout is a TTY; falls back to plain text otherwise.

        Args:
            dataset: A ParquetDataset or iterable yielding batches
            steps: Number of training steps (None = use total_kimg)
            log_every: Print metrics every N steps
            sample_every: Save sample images every N steps
            checkpoint_every: Save checkpoint every N steps
        """
        import sys
        from ..training.dataset import infinite_loader

        if steps is None:
            steps = (self.tcfg.total_kimg * 1000) // self.tcfg.batch_size

        loader = infinite_loader(dataset, self.tcfg.batch_size)
        t_start = time.time()

        # ── Try to load rich + halo ──────────────────────────────────────
        try:
            import halo as _halo
            from rich.progress import (
                Progress, BarColumn, TextColumn,
                TimeRemainingColumn, TimeElapsedColumn,
                SpinnerColumn, MofNCompleteColumn, TaskProgressColumn,
            )
            from rich.console import Console, Group
            from rich.live import Live
            from rich.rule import Rule
            _rich = True
        except ImportError:
            _rich = False

        if not _rich:
            # ── Plain-text fallback ──────────────────────────────────────
            print(f"\n{'='*60}\nTraining PixelGAN for {steps:,} steps\n{'='*60}")
            for step in range(1, steps + 1):
                batch = next(loader)
                metrics = self.train_on_batch(batch, step)
                self.cur_kimg += self.tcfg.batch_size / 1000.0
                if step % log_every == 0:
                    elapsed = time.time() - t_start
                    kimg_s = self.cur_kimg / elapsed
                    print(
                        f"Step {step:6d} | kimg {self.cur_kimg:6.1f} | "
                        f"G {metrics['g_loss']:.3f} | D {metrics['d_loss']:.3f} | "
                        f"R1 {metrics['r1']:.3f} | ADA {metrics['ada_p']:.3f} | "
                        f"{kimg_s:.2f} kimg/s"
                    )
                if step % sample_every == 0:
                    self._save_samples(step)
                    print(f"  Saved samples -> {self.output_dir}/samples_{step:06d}.png")
                if step % checkpoint_every == 0:
                    self._save_checkpoint(step)
                    print(f"  Saved checkpoint -> {self.output_dir}/checkpoints/step_{step:06d}")
            print(f"\nTraining complete! {self.cur_kimg:.1f} kimg processed")
            self._save_checkpoint(steps)
            self._save_samples(steps)
            return

        # ── Rich UI ──────────────────────────────────────────────────────
        console = Console(highlight=False)
        arch = self.arch
        console.print(Rule(
            f"[bold cyan]PixelGAN {arch.image_size}×{arch.image_size}  "
            f"· {self.tcfg.dataset_type}  ·  {steps:,} steps[/bold cyan]",
            style="cyan",
        ))

        # ── Step 1: JIT compilation spinner ─────────────────────────────
        sp = _halo.Halo(
            text=" Compiling JAX kernels… (takes a few minutes on first run)",
            spinner="dots12", color="cyan", stream=sys.stderr,
        )
        sp.start()
        t0 = time.time()
        batch = next(loader)
        metrics = self.train_on_batch(batch, 1)
        self.cur_kimg += self.tcfg.batch_size / 1000.0
        sp.succeed(f" Kernels compiled in {time.time() - t0:.1f}s")

        # ── Build multi-progress layout ─────────────────────────────────
        # Loss scale caps for bar fill (values above cap fill bar 100%)
        _G_CAP   = 3.0
        _D_CAP   = 3.0
        _R1_CAP  = 10.0
        _SPD_CAP = 0.3   # kimg/s

        # Main training bar (has ETA + elapsed)
        p_train = Progress(
            SpinnerColumn("dots2", style="cyan"),
            TextColumn("[bold cyan]Training  [/bold cyan]"),
            BarColumn(bar_width=40, complete_style="bold cyan",
                      finished_style="bold green"),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("[dim]ETA[/dim]"),
            TimeRemainingColumn(),
            TextColumn("{task.fields[speed]}", style="yellow"),
            console=console,
            expand=False,
        )
        t_train = p_train.add_task(
            "Training", total=steps, speed="",
        )
        p_train.advance(t_train, 1)  # step 1 already done

        # Metric gauge bars
        p_g   = self._make_bar_progress("  G loss   ", "bold blue")
        p_d   = self._make_bar_progress("  D loss   ", "bold magenta")
        p_r1  = self._make_bar_progress("  R1       ", "bold yellow")
        p_ada = self._make_bar_progress("  ADA p    ", "bold green")
        p_spd = self._make_bar_progress("  Speed    ", "bold bright_yellow")

        t_g   = p_g.add_task("g",   total=1000, val="—", start=False)
        t_d   = p_d.add_task("d",   total=1000, val="—", start=False)
        t_r1  = p_r1.add_task("r1", total=1000, val="—", start=False)
        t_ada = p_ada.add_task("ada", total=1000, val="—", start=False)
        t_spd = p_spd.add_task("spd", total=1000, val="—", start=False)

        def _refresh(step: int, m: dict, speed: float):
            g, d, r1, ada = m["g_loss"], m["d_loss"], m["r1"], m["ada_p"]
            p_train.update(t_train, completed=step,
                           speed=f"{speed:.3f} kimg/s")
            p_g.update(t_g,   completed=int(min(g   / _G_CAP,   1.0) * 1000),
                       val=f"[{'green' if 0.2 < g < 2.5 else 'red'}]{g:.4f}[/]")
            p_d.update(t_d,   completed=int(min(d   / _D_CAP,   1.0) * 1000),
                       val=f"[{'green' if 0.2 < d < 2.5 else 'red'}]{d:.4f}[/]")
            p_r1.update(t_r1,  completed=int(min(r1  / _R1_CAP,  1.0) * 1000),
                        val=f"{r1:.4f}")
            p_ada.update(t_ada, completed=int(min(ada, 1.0) * 1000),
                         val=f"[cyan]{ada:.3f}[/cyan]")
            p_spd.update(t_spd, completed=int(min(speed / _SPD_CAP, 1.0) * 1000),
                         val=f"[yellow]{speed:.3f} kimg/s[/yellow]")

        live_group = Group(p_train, p_g, p_d, p_r1, p_ada, p_spd)

        with Live(live_group, console=console, refresh_per_second=8):
            for step in range(2, steps + 1):
                batch = next(loader)
                metrics = self.train_on_batch(batch, step)
                self.cur_kimg += self.tcfg.batch_size / 1000.0

                if step % log_every == 0:
                    elapsed = time.time() - t_start
                    speed = self.cur_kimg / elapsed if elapsed > 0 else 0.0
                    _refresh(step, metrics, speed)

                if step % sample_every == 0:
                    path = self.output_dir / f"samples_{step:06d}.png"
                    self._save_samples(step)
                    console.print(
                        f"  [bold green]✓[/bold green] Saved samples   "
                        f"[dim]→ {path}[/dim]  [dim](step {step:,})[/dim]"
                    )

                if step % checkpoint_every == 0:
                    ckpt = self.output_dir / "checkpoints" / f"step_{step:06d}"
                    self._save_checkpoint(step)
                    console.print(
                        f"  [bold cyan]✓[/bold cyan]  Checkpoint saved "
                        f"[dim]→ {ckpt}[/dim]  [dim](step {step:,})[/dim]"
                    )

        console.print(Rule(
            f"[bold green]Training complete — {self.cur_kimg:.1f} kimg[/bold green]",
            style="green",
        ))
        self._save_checkpoint(steps)
        self._save_samples(steps)
