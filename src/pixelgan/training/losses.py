"""
PixelGAN loss functions.

All losses are implemented as pure JAX functions — no side effects,
compatible with jit, vmap, and grad.

Losses implemented:
  - Non-saturating GAN loss (StyleGAN2 style, softplus)
  - R1 gradient penalty (discriminator regularization)
  - Path length regularization (generator regularization)
  - Cycle consistency loss (for image-to-image mode)
  - L1 reconstruction loss (for image-to-image mode)
  - Palette quantization loss (encourage valid palette usage)

Key formulas (matching StyleGAN2 exactly):
  G loss:     softplus(-D(G(z)))
  D loss:     softplus(D(G(z))) + softplus(-D(x_real))
  R1 penalty: (1/2) * gamma * ||∇D(x_real)||^2
  PL penalty: (||∇_w [D(G(w)) * noise]||_2 - pl_mean)^2
"""

from __future__ import annotations

from functools import partial
from typing import Callable, Optional

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# GAN losses (non-saturating logistic, StyleGAN2 style)
# ---------------------------------------------------------------------------

def generator_loss(fake_logits: jnp.ndarray) -> jnp.ndarray:
    """
    Non-saturating generator loss.
    = -log(sigmoid(D(G(z))))
    = softplus(-D(G(z)))

    The generator tries to maximize D(G(z)) — make fake images look real.

    Args:
        fake_logits: [B] discriminator scores for fake images

    Returns:
        scalar loss
    """
    return jnp.mean(jax.nn.softplus(-fake_logits))


def discriminator_loss(
    real_logits: jnp.ndarray,
    fake_logits: jnp.ndarray,
) -> jnp.ndarray:
    """
    Non-saturating discriminator loss.
    = softplus(D(G(z))) + softplus(-D(x_real))
    = -log(sigmoid(D(x_real))) - log(1 - sigmoid(D(G(z))))

    Trains D to output high scores for real, low for fake.

    Args:
        real_logits: [B] D scores for real images
        fake_logits: [B] D scores for fake images

    Returns:
        scalar loss
    """
    loss_real = jnp.mean(jax.nn.softplus(-real_logits))
    loss_fake = jnp.mean(jax.nn.softplus(fake_logits))
    return loss_real + loss_fake


# ---------------------------------------------------------------------------
# Regularization
# ---------------------------------------------------------------------------

def r1_gradient_penalty(
    discriminator_fn: Callable,
    real_images: jnp.ndarray,   # [B, H, W, C]
    d_params: dict,
    r1_gamma: float = 10.0,
) -> jnp.ndarray:
    """
    R1 gradient penalty for discriminator regularization.

    Penalizes the gradient norm of the discriminator at real data points:
      loss_R1 = (gamma/2) * E[||∇_x D(x)||^2]

    This zero-centered gradient penalty stabilizes GAN training.
    Unlike WGAN-GP, R1 only penalizes at real points — simpler and effective.

    Args:
        discriminator_fn: Callable (x, params) → logits [B]
        real_images: [B, H, W, C] real training images
        d_params: Discriminator parameters
        r1_gamma: Regularization strength (10 is StyleGAN2 default)

    Returns:
        scalar R1 penalty
    """
    def d_real(images):
        return discriminator_fn(images, d_params)

    # Gradient of sum(D(x)) w.r.t. x
    grad_fn = jax.grad(lambda x: jnp.sum(d_real(x)))
    grads = grad_fn(real_images)  # [B, H, W, C]

    # R1 penalty: mean of squared gradient norms
    r1_penalty = jnp.sum(grads ** 2, axis=(1, 2, 3))  # [B]
    return (r1_gamma / 2.0) * jnp.mean(r1_penalty)


def path_length_regularization(
    generator_fn: Callable,
    ws: jnp.ndarray,       # [B, num_ws, w_dim] style codes
    g_params: dict,
    pl_mean: jnp.ndarray,  # scalar EMA of path length
    pl_weight: float = 2.0,
    pl_decay: float = 0.01,
    rng: Optional[jax.random.KeyArray] = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Path length regularization for generator.

    Encourages consistent mapping from W-space to image space:
    small moves in W → small moves in image space.

    PL penalty = (||J^T noise||_2 - pl_mean)^2
    where J is the Jacobian of G w.r.t. W.

    Args:
        generator_fn: Callable (ws, params) → images [B, H, W, C]
        ws: Style codes [B, num_ws, w_dim]
        g_params: Generator params
        pl_mean: Current EMA of path length (scalar)
        pl_weight: PL regularization weight
        pl_decay: EMA decay for pl_mean update
        rng: PRNG key

    Returns:
        (pl_penalty, new_pl_mean)
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)

    def gen_from_ws(w):
        return generator_fn(w, g_params)

    # Generate images and their shape
    images = gen_from_ws(ws)
    B, H, W, C = images.shape

    # Random direction in image space (unit variance noise)
    noise = jax.random.normal(rng, images.shape) / jnp.sqrt(H * W)

    # Compute J^T @ noise using vector-Jacobian product (VJP)
    _, vjp_fn = jax.vjp(gen_from_ws, ws)
    pl_grads = vjp_fn(noise)[0]  # [B, num_ws, w_dim]

    # Path lengths: ||J^T noise||_2 per sample
    pl_lengths = jnp.sqrt(
        jnp.sum(pl_grads ** 2, axis=(1, 2))  # sum over num_ws and w_dim
    )  # [B]

    # Update EMA
    pl_mean_batch = jnp.mean(pl_lengths)
    new_pl_mean = pl_mean + pl_decay * (pl_mean_batch - pl_mean)

    # PL penalty
    pl_penalty = jnp.mean((pl_lengths - new_pl_mean) ** 2) * pl_weight

    return pl_penalty, new_pl_mean


# ---------------------------------------------------------------------------
# Image-to-image losses
# ---------------------------------------------------------------------------

def cycle_consistency_loss(
    x_real: jnp.ndarray,   # [B, H, W, C] original image
    x_cycle: jnp.ndarray,  # [B, H, W, C] reconstructed: G_B(G_A(x))
    lambda_cycle: float = 10.0,
) -> jnp.ndarray:
    """
    Cycle consistency loss for unpaired image-to-image translation.

    Ensures G_B(G_A(x_A)) ≈ x_A and G_A(G_B(x_B)) ≈ x_B.
    Uses L1 norm (more robust than L2 for images).

    Args:
        x_real: Original images [B, H, W, C]
        x_cycle: Reconstructed images [B, H, W, C]
        lambda_cycle: Cycle loss weight (10 is CycleGAN default)

    Returns:
        scalar cycle consistency loss
    """
    return lambda_cycle * jnp.mean(jnp.abs(x_real - x_cycle))


def reconstruction_loss(
    x_real: jnp.ndarray,    # [B, H, W, C]
    x_pred: jnp.ndarray,    # [B, H, W, C]
    lambda_l1: float = 100.0,
    loss_type: str = "l1",
) -> jnp.ndarray:
    """
    Image reconstruction loss for paired image-to-image training (pix2pix style).

    Args:
        x_real: Target images [B, H, W, C]
        x_pred: Generated images [B, H, W, C]
        lambda_l1: L1 loss weight (100 is pix2pix default)
        loss_type: 'l1' or 'l2'

    Returns:
        scalar reconstruction loss
    """
    if loss_type == "l1":
        return lambda_l1 * jnp.mean(jnp.abs(x_real - x_pred))
    elif loss_type == "l2":
        return lambda_l1 * jnp.mean((x_real - x_pred) ** 2)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type!r}")


# ---------------------------------------------------------------------------
# Pixel-art-specific losses
# ---------------------------------------------------------------------------

def palette_coherence_loss(
    images: jnp.ndarray,     # [B, H, W, 3] RGB in [-1, 1]
    palette: jnp.ndarray,    # [N, 3] palette colors in [-1, 1]
    lambda_palette: float = 1.0,
) -> jnp.ndarray:
    """
    Palette coherence loss — encourage generated images to use palette colors.

    Computes the mean distance from each pixel to the nearest palette color.
    Differentiable via softmin approximation.

    This loss helps the model learn to "snap" to palette colors without
    hard quantization (which would break gradients).

    Args:
        images: [B, H, W, 3] generated images in [-1, 1]
        palette: [N, 3] palette colors in [-1, 1]
        lambda_palette: Loss weight

    Returns:
        scalar palette coherence loss
    """
    B, H, W, _ = images.shape
    N = palette.shape[0]

    # Expand for broadcasting: [B, H, W, 1, 3] - [1, 1, 1, N, 3] = [B, H, W, N, 3]
    pixels = images[:, :, :, None, :]   # [B, H, W, 1, 3]
    pal = palette[None, None, None, :, :]  # [1, 1, 1, N, 3]

    dists = jnp.sum((pixels - pal) ** 2, axis=-1)  # [B, H, W, N]

    # Soft-min distance to nearest palette color (differentiable)
    temp = 10.0  # temperature for softmin
    soft_min = -jax.nn.logsumexp(-dists * temp, axis=-1) / temp  # [B, H, W]

    return lambda_palette * jnp.mean(soft_min)


def total_variation_loss(
    images: jnp.ndarray,   # [B, H, W, C]
    lambda_tv: float = 0.1,
) -> jnp.ndarray:
    """
    Total variation loss — penalizes noise while preserving edges.

    For pixel art, we want a LOW weight here (0.01–0.1) — we WANT sharp edges.
    Used mainly to suppress salt-and-pepper noise in smooth areas.

    Args:
        images: [B, H, W, C] in [-1, 1]
        lambda_tv: TV loss weight (small for pixel art)

    Returns:
        scalar TV loss
    """
    # Differences between adjacent pixels
    diff_h = images[:, 1:, :, :] - images[:, :-1, :, :]  # [B, H-1, W, C]
    diff_w = images[:, :, 1:, :] - images[:, :, :-1, :]  # [B, H, W-1, C]

    return lambda_tv * (jnp.mean(jnp.abs(diff_h)) + jnp.mean(jnp.abs(diff_w)))


# ---------------------------------------------------------------------------
# Combined loss helpers
# ---------------------------------------------------------------------------

def compute_g_loss(
    fake_logits: jnp.ndarray,
    fake_images: jnp.ndarray,
    real_images: Optional[jnp.ndarray] = None,
    palette: Optional[jnp.ndarray] = None,
    lambda_recon: float = 0.0,
    lambda_palette: float = 0.0,
) -> dict[str, jnp.ndarray]:
    """
    Compute all generator losses, returning a dict.

    Args:
        fake_logits: [B] discriminator scores for fake images
        fake_images: [B, H, W, C] generated images
        real_images: [B, H, W, C] target images (for reconstruction loss)
        palette: [N, 3] palette colors (for palette loss)
        lambda_recon: Reconstruction loss weight (0 = disabled)
        lambda_palette: Palette loss weight (0 = disabled)

    Returns:
        dict with 'total', 'adversarial', 'reconstruction', 'palette' losses
    """
    losses = {}

    # Adversarial loss
    losses["adversarial"] = generator_loss(fake_logits)
    losses["total"] = losses["adversarial"]

    # Reconstruction loss (for img2img mode)
    if real_images is not None and lambda_recon > 0:
        losses["reconstruction"] = reconstruction_loss(
            real_images, fake_images, lambda_recon
        )
        losses["total"] = losses["total"] + losses["reconstruction"]

    # Palette coherence loss
    if palette is not None and lambda_palette > 0:
        losses["palette"] = palette_coherence_loss(
            fake_images[:, :, :, :3], palette, lambda_palette
        )
        losses["total"] = losses["total"] + losses["palette"]

    return losses


def compute_d_loss(
    real_logits: jnp.ndarray,
    fake_logits: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    """
    Compute discriminator losses.

    Args:
        real_logits: [B] scores for real images
        fake_logits: [B] scores for fake images

    Returns:
        dict with 'total', 'real', 'fake' losses
    """
    loss_real = jnp.mean(jax.nn.softplus(-real_logits))
    loss_fake = jnp.mean(jax.nn.softplus(fake_logits))

    return {
        "real": loss_real,
        "fake": loss_fake,
        "total": loss_real + loss_fake,
        # Heuristic: sign of real logits (should be > 0 for real)
        "real_sign": jnp.mean(jnp.sign(real_logits)),
    }
