"""
PixelGAN checkpoint utilities.

Simple pickle-based checkpointing with automatic versioning.
Saves: G params, G EMA params, D params, optimizer states, training counters.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np


def save_checkpoint(
    output_dir: str | Path,
    step: int,
    g_state,
    d_state,
    cur_kimg: float = 0.0,
    ada_p: float = 0.0,
    keep_last: int = 5,
) -> Path:
    """
    Save a training checkpoint.

    Args:
        output_dir: Base output directory
        step: Current training step
        g_state: Generator GANTrainState
        d_state: Discriminator TrainState
        cur_kimg: Current kimg counter
        ada_p: Current ADA probability
        keep_last: Number of recent checkpoints to keep

    Returns:
        Path to saved checkpoint directory
    """
    ckpt_dir = Path(output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / f"step_{step:07d}.pkl"

    import jax

    # Convert JAX arrays to numpy for pickling
    ckpt = {
        "step": step,
        "cur_kimg": cur_kimg,
        "ada_p": ada_p,
        "g_params": jax.tree_util.tree_map(np.array, g_state.params),
        "g_ema_params": jax.tree_util.tree_map(np.array, g_state.ema_params),
        "g_opt_state": jax.tree_util.tree_map(np.array, g_state.opt_state),
        "d_params": jax.tree_util.tree_map(np.array, d_state.params),
        "d_opt_state": jax.tree_util.tree_map(np.array, d_state.opt_state),
    }

    with open(ckpt_path, "wb") as f:
        pickle.dump(ckpt, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Clean up old checkpoints
    all_ckpts = sorted(ckpt_dir.glob("step_*.pkl"))
    if len(all_ckpts) > keep_last:
        for old in all_ckpts[:-keep_last]:
            old.unlink()

    return ckpt_path


def load_checkpoint(
    path: str | Path,
    g_state=None,
    d_state=None,
) -> dict:
    """
    Load a checkpoint.

    Args:
        path: Path to checkpoint .pkl file or directory (loads latest)
        g_state: Existing GANTrainState to update (optional)
        d_state: Existing TrainState to update (optional)

    Returns:
        dict with 'step', 'cur_kimg', 'ada_p', and updated states
    """
    path = Path(path)

    # If directory, find latest checkpoint
    if path.is_dir():
        ckpts = sorted(path.glob("checkpoints/step_*.pkl"))
        if not ckpts:
            ckpts = sorted(path.glob("step_*.pkl"))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints found in {path}")
        path = ckpts[-1]
        print(f"Loading latest checkpoint: {path}")

    with open(path, "rb") as f:
        ckpt = pickle.load(f)

    result = {
        "step": ckpt["step"],
        "cur_kimg": ckpt["cur_kimg"],
        "ada_p": ckpt["ada_p"],
    }

    # Convert numpy arrays back to JAX
    if g_state is not None:
        import jax
        import jax.numpy as jnp
        g_params = jax.tree_util.tree_map(jnp.array, ckpt["g_params"])
        g_ema = jax.tree_util.tree_map(jnp.array, ckpt["g_ema_params"])
        result["g_state"] = g_state.replace(params=g_params, ema_params=g_ema)

    if d_state is not None:
        import jax
        import jax.numpy as jnp
        d_params = jax.tree_util.tree_map(jnp.array, ckpt["d_params"])
        result["d_state"] = d_state.replace(params=d_params)

    return result


class CheckpointManager:
    """Manages checkpoint saving with configurable frequency."""

    def __init__(
        self,
        output_dir: str | Path,
        save_every: int = 2000,
        keep_last: int = 5,
    ):
        self.output_dir = Path(output_dir)
        self.save_every = save_every
        self.keep_last = keep_last
        self._last_saved = 0

    def maybe_save(
        self,
        step: int,
        g_state,
        d_state,
        cur_kimg: float,
        ada_p: float,
        force: bool = False,
    ) -> Optional[Path]:
        """Save if step % save_every == 0 or force=True."""
        if force or (step - self._last_saved >= self.save_every):
            path = save_checkpoint(
                self.output_dir, step, g_state, d_state,
                cur_kimg, ada_p, self.keep_last
            )
            self._last_saved = step
            return path
        return None
