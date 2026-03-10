"""
Option A: Palette-indexed output head for PixelGAN generator.

Instead of generating raw RGB floats, the generator outputs N logits per pixel.
A differentiable palette lookup converts these to RGB for the discriminator.
At inference, argmax gives the palette index directly.

Why this is powerful for pixel art:
  - Forces the model to pick from a legal colour set — no muddy intermediate hues
  - Separates the "what colour goes here" (classification) from "what shape" (synthesis)
  - The discriminator receives palette-constrained images, making it harder to fool
    without learning genuine sprite structure
  - Output is always a valid palette-indexed sprite — no post-processing quantisation

Architecture:
                         ┌─ train ─▶  softmax(logits) @ palette  ──▶ RGB [-1,1]
  synthesis feats ──▶ ToPaletteLogits
    [B,H,W,C]             [B,H,W,N]
                         └─ infer ─▶  argmax(logits) ──▶ uint8 index map

Palette encoding:
  - palette: float32 [N, 3] in [-1, 1]  (normalised same as model space)
  - Slot 0 is always the background / transparent colour
  - The model can therefore learn to output "transparent" for non-sprite pixels

Usage:
    head = ToPaletteLogits(n_colors=8, w_dim=128)
    logits = head(features, w_style)    # [B, H, W, 8]

    lookup = PaletteLookup()
    rgb = lookup(logits, palette)       # [B, H, W, 3] — differentiable

    # inference only:
    indices = decode_to_indices(logits) # [B, H, W] uint8
"""

from __future__ import annotations

from typing import Optional
import math

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np


# ---------------------------------------------------------------------------
# ToPaletteLogits — replaces ToRGB for palette-indexed mode
# ---------------------------------------------------------------------------

class ToPaletteLogits(nn.Module):
    """
    Converts final synthesis feature maps to per-pixel palette logits.

    Analogous to ToRGB from StyleGAN2, but outputs N class scores instead
    of 3 RGB channels. Uses the same 1×1 modulated convolution.

    Attributes:
        n_colors:  Number of palette entries (including slot 0 for transparent).
        w_dim:     Style code dimension.
        temperature: Softmax temperature during training. Lower = more
                     confident / sharper colour assignments.
    """
    n_colors:    int = 8
    w_dim:       int = 128
    temperature: float = 1.0

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,    # [B, H, W, C_in]
        w: jnp.ndarray,    # [B, w_dim]
    ) -> jnp.ndarray:      # [B, H, W, n_colors]  raw logits
        """
        Produce palette logits from synthesis features.

        Returns raw (un-normalised) logits. Apply PaletteLookup for training
        or decode_to_indices for inference.
        """
        from .generator import StyleAffine, modulated_conv2d  # local import avoids circular

        C_in = x.shape[-1]

        s = StyleAffine(C_in, name="affine")(w)
        weight_gain = 1.0 / math.sqrt(C_in)
        s = s * weight_gain

        kernel = self.param(
            "kernel",
            nn.initializers.normal(stddev=1.0),
            (1, 1, C_in, self.n_colors),
        )

        logits = modulated_conv2d(x, kernel, s, demodulate=False)

        # Bias: slot 0 (transparent / background) starts suppressed so the
        # generator prefers visible colours from the first step, not blank pages.
        def _init_bias(rng, shape):
            b = jnp.zeros(shape)
            return b.at[0].set(-2.0)   # suppress transparent slot at init

        logits = logits + self.param("bias", _init_bias, (self.n_colors,))
        return logits  # [B, H, W, n_colors]


# ---------------------------------------------------------------------------
# PaletteLookup — differentiable colour mapping (training)
# ---------------------------------------------------------------------------

class PaletteLookup(nn.Module):
    """
    Differentiable palette lookup: logits → RGB.

    Uses a softmax-weighted sum over palette colours:
        rgb = softmax(logits / temperature) @ palette   # [B, H, W, 3]

    This is fully differentiable and allows gradients to flow back into
    the generator's logit predictions.

    Temperature schedule:
        epoch 0-25%:  tau = 1.0  (soft / exploratory)
        epoch 25-75%: tau = 0.3  (sharpening towards argmax)
        epoch 75-100%: tau = 0.1 (near-argmax commitment)

    In practice the trainer can anneal `temperature` as training progresses.
    """
    temperature: float = 1.0

    @nn.compact
    def __call__(
        self,
        logits:  jnp.ndarray,   # [B, H, W, N]
        palette: jnp.ndarray,   # [N, 3] or [B, N, 3]  float32 in [-1, 1]
        temperature: Optional[float] = None,
    ) -> jnp.ndarray:           # [B, H, W, 3]
        """
        Convert palette logits to RGB via soft colour mixing.

        Args:
            logits:      Per-pixel palette logits from ToPaletteLogits.
            palette:     float32 [N, 3] (shared) or [B, N, 3] (per-sample).
            temperature: Override the module's temperature for this call.

        Returns:
            float32 [B, H, W, 3] RGB in [-1, 1].
        """
        tau = temperature if temperature is not None else self.temperature
        probs = jax.nn.softmax(logits / tau, axis=-1)  # [B, H, W, N]
        if palette.ndim == 2:
            # Single palette broadcast across all batch samples
            rgb = jnp.einsum("bhwn,nc->bhwc", probs, palette)
        else:
            # Per-sample palettes: [B, N, 3]
            rgb = jnp.einsum("bhwn,bnc->bhwc", probs, palette)
        return rgb


# ---------------------------------------------------------------------------
# palette_lookup — free function version (no nn.Module overhead)
# ---------------------------------------------------------------------------

def palette_lookup(
    logits:      jnp.ndarray,          # [B, H, W, N]
    palette:     jnp.ndarray,          # [N, 3] or [B, N, 3]  float32 in [-1,1]
    temperature: float = 1.0,
) -> jnp.ndarray:                      # [B, H, W, 3]
    """
    Differentiable palette lookup: logits → RGB via softmax-weighted sum.

    Drop-in replacement for ``PaletteLookup()(logits, palette, temperature)``
    that works outside of Flax modules (e.g. inside jit-compiled functions).
    """
    probs = jax.nn.softmax(logits / temperature, axis=-1)  # [B, H, W, N]
    if palette.ndim == 2:
        rgb = jnp.einsum("bhwn,nc->bhwc", probs, palette)
    else:
        rgb = jnp.einsum("bhwn,bnc->bhwc", probs, palette)
    return rgb


# ---------------------------------------------------------------------------
# Hard (argmax) decode — inference only, not differentiable
# ---------------------------------------------------------------------------

def decode_to_indices(logits: jnp.ndarray) -> jnp.ndarray:
    """
    Hard decode: logits → palette indices via argmax.
    Output is NOT differentiable; use only for inference / sample saving.

    Args:
        logits: [B, H, W, N] float32 palette logits.

    Returns:
        [B, H, W] uint8 palette index per pixel.
    """
    return jnp.argmax(logits, axis=-1).astype(jnp.uint8)


def indices_to_rgb_numpy(
    indices: np.ndarray,
    palette: np.ndarray,
    bg_rgb:  tuple[int, int, int] = (40, 40, 40),
) -> np.ndarray:
    """
    Convert uint8 index maps to uint8 RGB images (numpy, CPU-only).

    Args:
        indices: [B, H, W] or [H, W] uint8 palette indices.
        palette: [N, 3] uint8 palette (slot 0 = transparent/bg).
        bg_rgb:  Background colour for transparent pixels (index 0).

    Returns:
        uint8 [B, H, W, 3] or [H, W, 3].
    """
    batched = indices.ndim == 3
    if not batched:
        indices = indices[None]

    B, H, W = indices.shape
    out = np.empty((B, H, W, 3), dtype=np.uint8)
    for b in range(B):
        mask_transparent = indices[b] == 0
        idx_clipped = np.clip(indices[b], 0, palette.shape[0] - 1)
        out[b] = palette[idx_clipped]
        out[b][mask_transparent] = bg_rgb  # override bg

    return out if batched else out[0]


# ---------------------------------------------------------------------------
# Palette temperature schedule
# ---------------------------------------------------------------------------

def get_palette_temperature(
    step: int,
    total_steps: int,
    t_start: float = 0.5,
    t_end:   float = 0.05,
) -> float:
    """
    Cosine annealing schedule for palette softmax temperature.

    Starts moderately soft (t_start=0.5, enough contrast to see structure),
    ends very sharp (t_end=0.05, near-argmax hard assignments).

    t_start was previously 1.0, which kept softmax nearly uniform for the
    first ~25% of training and caused all generated images to look like
    colour-averaged blobs.  0.5 gives enough initial contrast for D to see
    structure while still being differentiable.

    Args:
        step:        Current training step.
        total_steps: Total training steps.
        t_start:     Starting temperature (soft).
        t_end:       Final temperature (sharp, close to argmax).

    Returns:
        float temperature for this step.
    """
    progress = min(step / max(total_steps, 1), 1.0)
    # Cosine decay from t_start to t_end
    cos_factor = (1 + math.cos(math.pi * progress)) / 2.0
    return t_end + (t_start - t_end) * cos_factor
