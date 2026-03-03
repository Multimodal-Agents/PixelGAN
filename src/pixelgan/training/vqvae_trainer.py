"""
VQ-VAE Stage 1 Trainer.

Trains the Vector-Quantized Variational Autoencoder (Option C) independently
of the GAN. The trained encoder/decoder weights are saved and later used by
the Stage 2 GAN training, where the GAN generator operates in the compact
8×8 latent space and the frozen VQ-VAE decoder expands it back to 64×64.

Typical usage (via train_vqvae.py script):
    config = get_config(64)
    config.vqvae = VQVAEConfig(codebook_size=256, latent_dim=64)
    trainer = VQVAETrainer(config, output_dir="runs/vqvae")
    dataset = load_dataset("datasets/sprites.parquet", "seed", image_size=64)
    trainer.fit(dataset, steps=10_000)
    # → saves checkpoint to runs/vqvae/checkpoint/

Training regime:
  - Adam optimiser, lr=1e-3, cosine decay
  - Reconstruction loss: 0.5×L1 + 0.5×L2
  - VQ commitment + codebook loss (straight-through)
  - Snapshots: reconstructions saved as PNG grids every snapshot_steps
  - Orbax checkpoint every snapshot_steps
"""

from __future__ import annotations

import os
import time
from functools import partial
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

from ..models.vqvae import VQVAE, make_vqvae, vqvae_loss
from ..training.dataset import ParquetDataset, infinite_loader
from ..utils.config import PixelGANConfig, VQVAEConfig


# ---------------------------------------------------------------------------
# Training state
# ---------------------------------------------------------------------------

class VQVAETrainState(train_state.TrainState):
    """VQ-VAE train state (standard; VQ bottleneck has no mutable state here)."""
    pass


# ---------------------------------------------------------------------------
# JIT-compiled training step
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnums=(0,))
def train_step_vqvae(
    model:         VQVAE,
    state:         VQVAETrainState,
    images:        jnp.ndarray,        # [B, H, W, 3] float32 [-1, 1]
    lambda_recon:  float,
    lambda_vq:     float,
) -> tuple[VQVAETrainState, dict]:
    """
    Single VQ-VAE training step.

    Returns:
        updated state, metrics dict
    """
    def loss_fn(params):
        x_recon, indices, vq_loss_val = model.apply(
            {"params": params},
            images,
            train=True,
        )
        total, metrics = vqvae_loss(
            images, x_recon, vq_loss_val,
            lambda_recon=lambda_recon,
            lambda_vq=lambda_vq,
        )
        metrics["n_unique_codes"] = jnp.unique(indices).shape[0]
        return total, metrics

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        state.params
    )
    state = state.apply_gradients(grads=grads)
    return state, metrics


@partial(jax.jit, static_argnums=(0,))
def eval_step_vqvae(
    model:  VQVAE,
    params: dict,
    images: jnp.ndarray,
) -> jnp.ndarray:
    """Reconstruct images for visual inspection (uses greedy argmax quantization)."""
    x_recon, _, _ = model.apply({"params": params}, images, train=False)
    return x_recon


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class VQVAETrainer:
    """
    Stage 1 trainer for the VQ-VAE.

    After training, saves orbax checkpoint to ``output_dir/checkpoint/`` plus
    a ``decoder_only/`` sub-checkpoint for use in Stage 2 GAN training.

    Args:
        config:     PixelGANConfig with a non-None ``vqvae`` field.
        output_dir: Save directory for logs, samples, and checkpoints.
    """

    FIXED_EVAL_BATCH = 16  # Number of samples to reconstruct for visual log

    def __init__(
        self,
        config:     PixelGANConfig,
        output_dir: Optional[str] = None,
    ):
        if config.vqvae is None:
            raise ValueError(
                "PixelGANConfig.vqvae must be set to a VQVAEConfig instance. "
                "Example: config.vqvae = VQVAEConfig()"
            )

        self.config    = config
        self.vcfg: VQVAEConfig = config.vqvae
        self.image_size = config.arch.image_size

        self.output_dir = Path(output_dir or self.vcfg.checkpoint_path).parent
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir = self.output_dir / "samples"
        self.sample_dir.mkdir(exist_ok=True)
        self.ckpt_dir = self.output_dir / "checkpoint"
        self.ckpt_dir.mkdir(exist_ok=True)

        # Build model
        self.model: VQVAE = make_vqvae(config)

        # Init state (deferred until fit() to know the input shape)
        self.state: Optional[VQVAETrainState] = None

        # Metrics history
        self.history: list[dict] = []

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _build_state(self, sample_image: jnp.ndarray) -> VQVAETrainState:
        """Initialise model and optimiser from a sample input."""
        vcfg = self.vcfg

        # Cosine-decayed learning rate
        schedule = optax.cosine_decay_schedule(
            init_value=vcfg.lr,
            decay_steps=vcfg.total_steps,
        )
        tx = optax.adam(learning_rate=schedule)

        rng = jax.random.PRNGKey(self.config.training.seed)
        variables = self.model.init(rng, sample_image, train=False)

        n_params = sum(p.size for p in jax.tree_util.tree_leaves(variables["params"]))
        print(f"  VQ-VAE parameters: {n_params:,}")
        print(
            f"  Codebook: {vcfg.codebook_size} codes × {vcfg.latent_dim} dims"
            f"  (latent grid: {self.image_size // 8}×{self.image_size // 8})"
        )

        return VQVAETrainState.create(
            apply_fn=self.model.apply,
            params=variables["params"],
            tx=tx,
        )

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _save_checkpoint(self, step: int) -> None:
        """Save model weights to orbax checkpoint."""
        try:
            import orbax.checkpoint as ocp

            checkpointer = ocp.StandardCheckpointer()
            ckpt = {"params": self.state.params, "step": step}
            checkpointer.save(str(self.ckpt_dir / f"step_{step:07d}"), ckpt)

            # Always overwrite 'latest' symlink-style by saving to fixed path
            checkpointer.save(str(self.ckpt_dir / "latest"), ckpt)

        except ImportError:
            # Fall back to numpy if orbax not available
            np.savez(
                self.ckpt_dir / f"params_step{step:07d}.npz",
                **{k: np.array(v) for k, v in
                   jax.tree_util.tree_leaves_with_path(self.state.params)},
            )

    def load_checkpoint(self, ckpt_path: Optional[str] = None) -> None:
        """Load weights from checkpoint (for resuming or Stage 2)."""
        try:
            import orbax.checkpoint as ocp

            path = ckpt_path or str(self.ckpt_dir / "latest")
            checkpointer = ocp.StandardCheckpointer()
            restored = checkpointer.restore(path)
            self.state = self.state.replace(params=restored["params"])
            print(f"  Loaded checkpoint from {path}")

        except ImportError:
            raise RuntimeError(
                "orbax-checkpoint required to load checkpoints. "
                "Install: pip install orbax-checkpoint"
            )

    # ------------------------------------------------------------------
    # Sample saving
    # ------------------------------------------------------------------

    def _save_reconstructions(self, step: int, real: np.ndarray) -> None:
        """Save a grid of real vs reconstructed images as PNG."""
        from PIL import Image as PILImage

        recon = np.array(eval_step_vqvae(self.model, self.state.params, jnp.array(real)))

        def to_uint8(x):
            return ((np.clip(x, -1, 1) + 1) * 127.5).astype(np.uint8)

        real_u8  = to_uint8(real[:8])
        recon_u8 = to_uint8(recon[:8])

        # Tile: top row = real, bottom row = reconstructed
        H, W, C = real_u8[0].shape
        n = 8
        grid = np.zeros((H * 2, W * n, C), dtype=np.uint8)
        for i in range(n):
            grid[:H, i * W:(i + 1) * W] = real_u8[i]
            grid[H:,  i * W:(i + 1) * W] = recon_u8[i]

        mode = "RGB" if C == 3 else "RGBA"
        PILImage.fromarray(grid, mode).save(
            self.sample_dir / f"recon_step{step:07d}.png"
        )

    # ------------------------------------------------------------------
    # Codebook utilization diagnostics
    # ------------------------------------------------------------------

    def _log_codebook_utilization(self, dataset: ParquetDataset) -> None:
        """Count how many unique codes are actually used across the dataset."""
        used = set()
        n_check = min(len(dataset), 512)
        batch = dataset.get_batch(n_check, 0)
        images = jnp.array(batch["image"])

        # Encode in chunks to avoid OOM
        chunk = 16
        for i in range(0, n_check, chunk):
            x_chunk = images[i:i + chunk]
            _, indices, _ = self.model.apply(
                {"params": self.state.params}, x_chunk, train=False
            )
            used.update(int(x) for x in np.array(indices).ravel())

        K = self.vcfg.codebook_size
        print(f"  Codebook utilisation: {len(used)}/{K} codes used "
              f"({100 * len(used) / K:.1f}%)")

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def fit(
        self,
        dataset: ParquetDataset,
        steps:   Optional[int] = None,
        start_step: int = 0,
    ) -> dict:
        """
        Train the VQ-VAE for ``steps`` gradient steps.

        Args:
            dataset:     A ParquetDataset (SeedDataset, TextDataset, etc.)
            steps:       Number of gradient steps. Defaults to vcfg.total_steps.
            start_step:  Resume training from this step (for resuming).

        Returns:
            Metrics dict (averages over last 100 steps).
        """
        vcfg  = self.vcfg
        steps = steps or vcfg.total_steps

        # Build state from a sample batch
        sample_batch = dataset.get_batch(2)
        sample_img   = jnp.array(sample_batch["image"][:1])

        if self.state is None:
            print("\n[VQ-VAE] Initialising model...")
            self.state = self._build_state(sample_img)

        # Keep a fixed eval batch for visual logs
        eval_batch  = dataset.get_batch(self.FIXED_EVAL_BATCH)
        eval_images = np.array(eval_batch["image"])

        print(f"\n[VQ-VAE] Training for {steps:,} steps on {len(dataset):,} samples")
        print(f"  Output dir: {self.output_dir}")

        loader = infinite_loader(dataset, vcfg.batch_size, shuffle=True)
        t0 = time.time()

        for step in range(start_step, start_step + steps):
            batch = next(loader)
            images_np = batch["image"]
            # Ensure 3-channel (drop alpha if present)
            if images_np.shape[-1] == 4:
                images_np = images_np[..., :3]

            images_jax = jnp.array(images_np)

            self.state, metrics = train_step_vqvae(
                self.model,
                self.state,
                images_jax,
                lambda_recon=vcfg.lambda_recon,
                lambda_vq=vcfg.lambda_vq,
            )

            self.history.append({"step": step, **metrics})

            # ----------------------------------------------------------
            # Logging
            # ----------------------------------------------------------
            if step % 100 == 0 or step == start_step + steps - 1:
                elapsed = time.time() - t0
                fps     = (step - start_step + 1) * vcfg.batch_size / max(elapsed, 1e-6)
                print(
                    f"  step {step:6d}/{start_step + steps - 1}  "
                    f"loss={metrics['total']:.4f}  "
                    f"recon_l1={metrics['recon_l1']:.4f}  "
                    f"vq={metrics['vq_loss']:.4f}  "
                    f"codes={int(metrics['n_unique_codes'])}  "
                    f"({fps:.1f} img/s)"
                )

            # ----------------------------------------------------------
            # Snapshots
            # ----------------------------------------------------------
            if (step + 1) % vcfg.snapshot_steps == 0:
                self._save_reconstructions(step + 1, eval_images)
                self._save_checkpoint(step + 1)
                self._log_codebook_utilization(dataset)

        # Final checkpoint + utilization
        self._save_checkpoint(start_step + steps)
        self._log_codebook_utilization(dataset)
        self._save_reconstructions(start_step + steps, eval_images)

        # Return average of last 100 steps
        tail = self.history[-100:]
        avg = {
            k: float(np.mean([m[k] for m in tail]))
            for k in tail[0] if k != "step"
        }
        print(f"\n[VQ-VAE] Training complete. Average (last 100 steps): {avg}")
        return avg


# ---------------------------------------------------------------------------
# Convenience: load a trained decoder for Stage 2 GAN
# ---------------------------------------------------------------------------

def load_vqvae_decoder(
    config:   PixelGANConfig,
    ckpt_path: str,
) -> tuple[VQVAE, dict]:
    """
    Load a trained VQ-VAE from checkpoint and return (model, params).

    The GAN Stage 2 uses the decoder standalone:
        x_64x64 = model.decode(params, z_8x8)

    Args:
        config:    PixelGANConfig with matching vqvae sub-config.
        ckpt_path: Path to checkpoint directory (contains 'latest/').

    Returns:
        (VQVAE model, params dict)
    """
    try:
        import orbax.checkpoint as ocp
    except ImportError:
        raise RuntimeError("orbax-checkpoint required: pip install orbax-checkpoint")

    model = make_vqvae(config)
    checkpointer = ocp.StandardCheckpointer()
    restored = checkpointer.restore(ckpt_path)
    return model, restored["params"]
