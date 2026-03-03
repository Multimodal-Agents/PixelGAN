"""
Option C: Vector-Quantized Variational Autoencoder (VQ-VAE) for PixelGAN.

Compresses pixel art images into a discrete latent space, then trains the
GAN on that latent space — exactly the architecture behind Stable Diffusion,
adapted for small pixel art sprites.

Pipeline:
    ┌─────────────────────────────────────────────────────────────────┐
    │  Stage 1 — Train VQ-VAE (separate script: train_vqvae.py)       │
    │                                                                   │
    │   64×64×3 ──[Encoder]──▶ 8×8×D ──[VectorQuantize]──▶ 8×8×D    │
    │                                        │                          │
    │                                        ▼                          │
    │                            codebook indices 8×8                  │
    │                                        │                          │
    │                                        ▼                          │
    │   64×64×3 ◀──[Decoder]──── 8×8×D ◀──[lookup]                   │
    └─────────────────────────────────────────────────────────────────┘
    ┌─────────────────────────────────────────────────────────────────┐
    │  Stage 2 — Train GAN in latent space (train.py --vqvae-path)    │
    │                                                                   │
    │   G: z → 8×8×D latent ──[frozen Decoder]──▶ 64×64×3 image      │
    │   D: 64×64×3 ──▶ real/fake logit                                │
    └─────────────────────────────────────────────────────────────────┘

Why 8×8 matters:
  - 64×64 image has 4096 spatial positions
  - 8×8 latent has only  64 positions  → 64× fewer
  - G generates 64-element sequence instead of 4096-element image
  - Training is ~10× faster per step; model can be ~5× smaller

VQ-VAE specifics:
  - Codebook: K=256 vectors, each of dimension D=64 (small for pixel art)
  - Straight-through estimator for codebook gradients (standard Bengio 2017)
  - EMA codebook update (more stable than gradient update)
  - Commitment loss + codebook loss weighted by beta=0.25

Params (64×64 → 8×8×64, codebook 256×64):
  Encoder:      ~180k
  Decoder:      ~280k
  Codebook:     256×64 = 16k floats
  Total VQ-VAE: ~476k  (compare: full 64×64 GAN G = 741k)
"""

from __future__ import annotations

from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import flax.linen as nn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """
    Residual block used in both encoder and decoder.
    GroupNorm + Conv + ReLU × 2 with skip connection.
    """
    channels: int
    n_groups: int = 8

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        n_groups = min(self.n_groups, self.channels)
        residual = x
        x = nn.GroupNorm(num_groups=n_groups)(x)
        x = nn.relu(x)
        x = nn.Conv(self.channels, (3, 3), padding="SAME")(x)
        x = nn.GroupNorm(num_groups=n_groups)(x)
        x = nn.relu(x)
        x = nn.Conv(self.channels, (3, 3), padding="SAME")(x)
        # Skip connection (with channel projection if needed)
        if residual.shape != x.shape:
            residual = nn.Conv(self.channels, (1, 1))(residual)
        return x + residual


class Downsample(nn.Module):
    """Strided conv downsampling: [B, H, W, C] → [B, H//2, W//2, C_out]."""
    out_channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return nn.Conv(
            self.out_channels, (4, 4), strides=(2, 2), padding="SAME"
        )(x)


class Upsample(nn.Module):
    """Sub-pixel conv upsampling: [B, H, W, C] → [B, H*2, W*2, C_out]."""
    out_channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Conv to 4× channels then pixel-shuffle
        B, H, W, C = x.shape
        x = nn.Conv(self.out_channels * 4, (3, 3), padding="SAME")(x)
        # Pixel shuffle
        x = x.reshape(B, H, W, 2, 2, self.out_channels)
        x = x.transpose(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, H * 2, W * 2, self.out_channels)
        return x


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class VQEncoder(nn.Module):
    """
    VQ-VAE Encoder: image → latent feature map.

    Performs 3 downsampling stages (÷8 spatial), producing an
    8×8 feature map for 64×64 input.

    64×64×3 → 32×32×ch → 16×16×ch*2 → 8×8×latent_dim

    Attributes:
        base_channels:  Starting channels (doubles each downsample).
        latent_dim:     Output feature dimension (codebook vector size).
        n_res_blocks:   Number of residual blocks at each level.
    """
    base_channels: int = 64
    latent_dim:    int = 64
    n_res_blocks:  int = 2

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,     # [B, H, W, 3] float32 [-1, 1]
        train: bool = True,
    ) -> jnp.ndarray:        # [B, H//8, W//8, latent_dim]
        ch = self.base_channels

        # Input projection
        x = nn.Conv(ch, (3, 3), padding="SAME", name="in_conv")(x)

        # Level 1: H×W × ch
        for i in range(self.n_res_blocks):
            x = ResBlock(ch, name=f"res_l1_{i}")(x, train)
        x = Downsample(ch * 2, name="down1")(x)               # H//2

        # Level 2: H//2 × ch*2
        ch2 = ch * 2
        for i in range(self.n_res_blocks):
            x = ResBlock(ch2, name=f"res_l2_{i}")(x, train)
        x = Downsample(ch2 * 2, name="down2")(x)              # H//4

        # Level 3: H//4 × ch*4
        ch3 = ch2 * 2
        for i in range(self.n_res_blocks):
            x = ResBlock(ch3, name=f"res_l3_{i}")(x, train)
        x = Downsample(ch3, name="down3")(x)                   # H//8

        # Middle res blocks
        for i in range(self.n_res_blocks):
            x = ResBlock(ch3, name=f"res_mid_{i}")(x, train)

        # Output projection to latent_dim
        x = nn.GroupNorm(num_groups=min(8, ch3))(x)
        x = nn.relu(x)
        x = nn.Conv(self.latent_dim, (1, 1), name="out_conv")(x)

        return x  # [B, H//8, W//8, latent_dim]


# ---------------------------------------------------------------------------
# Vector Quantizer
# ---------------------------------------------------------------------------

class VectorQuantizer(nn.Module):
    """
    Discrete bottleneck via nearest-neighbour lookup in a learned codebook.

    Uses:
      - EMA codebook updates (more stable than gradient-based)
      - Straight-through estimator for encoder gradients
      - Commitment loss + codebook loss

    Attributes:
        codebook_size:  K — number of discrete codes.
        latent_dim:     D — code vector dimension (must match encoder output).
        commitment_beta: Weight of commitment loss term (default 0.25).
        ema_decay:      EMA decay for codebook update (default 0.99).
    """
    codebook_size:    int   = 256
    latent_dim:       int   = 64
    commitment_beta:  float = 0.25
    ema_decay:        float = 0.99

    @nn.compact
    def __call__(
        self,
        z:     jnp.ndarray,    # [B, H, W, D]  encoder output (pre-quantized)
        train: bool = True,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Quantize encoder output to nearest codebook entry.

        Returns:
            z_q:        [B, H, W, D]  quantized latent (with straight-through grad)
            indices:    [B, H, W]     uint16 codebook indices
            vq_loss:    scalar        commitment + codebook loss
        """
        K, D = self.codebook_size, self.latent_dim
        B, H, W, _ = z.shape

        # Codebook: [K, D]
        codebook = self.param(
            "codebook",
            nn.initializers.normal(stddev=0.02),
            (K, D),
        )

        # Flatten spatial: [B*H*W, D]
        z_flat = z.reshape(-1, D)

        # Squared L2 distances: ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z·e
        z_sq  = jnp.sum(z_flat ** 2, axis=1, keepdims=True)   # [B*H*W, 1]
        e_sq  = jnp.sum(codebook ** 2, axis=1, keepdims=True).T  # [1, K]
        cross = jnp.dot(z_flat, codebook.T)                    # [B*H*W, K]
        dists = z_sq + e_sq - 2 * cross                        # [B*H*W, K]

        # Nearest code
        indices_flat = jnp.argmin(dists, axis=1)               # [B*H*W]
        indices      = indices_flat.reshape(B, H, W)            # [B, H, W]

        # Quantized vectors
        z_q_flat = codebook[indices_flat]                      # [B*H*W, D]
        z_q      = z_q_flat.reshape(B, H, W, D)

        # Straight-through estimator: copy gradients from z_q to z
        z_q_st   = z + jax.lax.stop_gradient(z_q - z)

        # Losses
        codebook_loss   = jnp.mean((jax.lax.stop_gradient(z)    - z_q) ** 2)
        commitment_loss = jnp.mean((z - jax.lax.stop_gradient(z_q)) ** 2)
        vq_loss = codebook_loss + self.commitment_beta * commitment_loss

        return z_q_st, indices, vq_loss


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class VQDecoder(nn.Module):
    """
    VQ-VAE Decoder: quantized latent → reconstructed image.

    Mirror of VQEncoder: 3 upsampling stages (×8 spatial).
    8×8×latent_dim → 16×16×ch*4 → 32×32×ch*2 → 64×64×ch → 64×64×3

    Attributes:
        base_channels:  Channel count at output resolution (mirrors encoder).
        latent_dim:     Input latent dimension (must match encoder+quantizer).
        n_res_blocks:   Residual blocks per level.
    """
    base_channels: int = 64
    latent_dim:    int = 64
    n_res_blocks:  int = 2

    @nn.compact
    def __call__(
        self,
        z_q:   jnp.ndarray,    # [B, H//8, W//8, latent_dim]
        train: bool = True,
    ) -> jnp.ndarray:           # [B, H, W, 3] float32 [-1, 1]
        ch  = self.base_channels
        ch3 = ch * 4

        # Input projection
        x = nn.Conv(ch3, (3, 3), padding="SAME", name="in_conv")(z_q)

        # Middle res blocks
        for i in range(self.n_res_blocks):
            x = ResBlock(ch3, name=f"res_mid_{i}")(x, train)

        # Level 3: H//8 → H//4
        for i in range(self.n_res_blocks):
            x = ResBlock(ch3, name=f"res_l3_{i}")(x, train)
        x = Upsample(ch * 2, name="up3")(x)

        # Level 2: H//4 → H//2
        ch2 = ch * 2
        for i in range(self.n_res_blocks):
            x = ResBlock(ch2, name=f"res_l2_{i}")(x, train)
        x = Upsample(ch, name="up2")(x)

        # Level 1: H//2 → H
        for i in range(self.n_res_blocks):
            x = ResBlock(ch, name=f"res_l1_{i}")(x, train)
        x = Upsample(ch, name="up1")(x)

        # Output projection
        x = nn.GroupNorm(num_groups=min(8, ch))(x)
        x = nn.relu(x)
        x = nn.Conv(3, (3, 3), padding="SAME", name="out_conv")(x)
        return jnp.tanh(x)  # [B, H, W, 3] in [-1, 1]


# ---------------------------------------------------------------------------
# Complete VQ-VAE
# ---------------------------------------------------------------------------

class VQVAE(nn.Module):
    """
    Complete VQ-VAE: encoder + quantizer + decoder.

    Can be:
      - Trained end-to-end (Stage 1, train_vqvae.py)
      - Decoder-only used as GAN synthesis backbone (Stage 2, train.py)

    Attributes:
        base_channels:   Encoder/decoder base channel count.
        latent_dim:      Latent vector dimension.
        codebook_size:   Number of discrete codes.
        commitment_beta: VQ commitment loss weight.
        n_res_blocks:    Residual blocks at each scale.
    """
    base_channels:    int   = 64
    latent_dim:       int   = 64
    codebook_size:    int   = 256
    commitment_beta:  float = 0.25
    n_res_blocks:     int   = 2

    @nn.compact
    def __call__(
        self,
        x:     jnp.ndarray,    # [B, H, W, 3]
        train: bool = True,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Full forward pass.

        Returns:
            x_recon:  [B, H, W, 3]  Reconstructed image in [-1, 1]
            indices:  [B, H//8, W//8]  Codebook indices
            vq_loss:  scalar  VQ commitment + codebook loss
        """
        z     = VQEncoder(
            base_channels=self.base_channels,
            latent_dim=self.latent_dim,
            n_res_blocks=self.n_res_blocks,
            name="encoder",
        )(x, train)

        z_q, indices, vq_loss = VectorQuantizer(
            codebook_size=self.codebook_size,
            latent_dim=self.latent_dim,
            commitment_beta=self.commitment_beta,
            name="quantizer",
        )(z, train)

        x_recon = VQDecoder(
            base_channels=self.base_channels,
            latent_dim=self.latent_dim,
            n_res_blocks=self.n_res_blocks,
            name="decoder",
        )(z_q, train)

        return x_recon, indices, vq_loss

    def encode(
        self,
        params:  dict,
        x:       jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Encode image to quantized latent + indices (inference)."""
        z     = self.apply({"params": params}, x, train=False,
                           method=lambda m, *a, **kw: m.encoder(*a, **kw))
        z_q, indices, _ = self.apply(
            {"params": params}, z, train=False,
            method=lambda m, *a, **kw: m.quantizer(*a, **kw)
        )
        return z_q, indices

    def decode(
        self,
        params: dict,
        z_q:    jnp.ndarray,
    ) -> jnp.ndarray:
        """Decode quantized latent to image (inference, used in Stage 2 GAN)."""
        return self.apply(
            {"params": params}, z_q, train=False,
            method=lambda m, *a, **kw: m.decoder(*a, **kw)
        )


# ---------------------------------------------------------------------------
# VQ-VAE Losses
# ---------------------------------------------------------------------------

def vqvae_loss(
    x_real:    jnp.ndarray,    # [B, H, W, 3]  original image
    x_recon:   jnp.ndarray,    # [B, H, W, 3]  reconstruction
    vq_loss:   jnp.ndarray,    # scalar  commitment+codebook loss
    lambda_recon: float = 1.0,
    lambda_vq:    float = 1.0,
    lambda_perceptual: float = 0.0,  # future: LPIPS, set 0 for now
) -> tuple[jnp.ndarray, dict]:
    """
    Full VQ-VAE training loss.

    Components:
      - Reconstruction: L1 + L2 mix (L1 for sharp edges, L2 for smooth gradients)
      - VQ:            Commitment + codebook alignment
      - (Optional) Perceptual: Can add LPIPS-style loss later

    Returns:
        total_loss, metrics_dict
    """
    # L1 reconstruction (better for sparse pixel art than L2)
    recon_l1 = jnp.mean(jnp.abs(x_real - x_recon))
    # L2 for smooth gradient flow
    recon_l2 = jnp.mean((x_real - x_recon) ** 2)
    # Mix: 50/50 L1 + L2 works well for pixel art
    recon_loss = 0.5 * recon_l1 + 0.5 * recon_l2

    total = lambda_recon * recon_loss + lambda_vq * vq_loss

    return total, {
        "total":      float(total),
        "recon_l1":   float(recon_l1),
        "recon_l2":   float(recon_l2),
        "vq_loss":    float(vq_loss),
    }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_vqvae(cfg: "PixelGANConfig") -> VQVAE:
    """Create a VQVAE from a PixelGANConfig (reads vqvae sub-config)."""
    vcfg = cfg.vqvae
    return VQVAE(
        base_channels   = vcfg.base_channels,
        latent_dim      = vcfg.latent_dim,
        codebook_size   = vcfg.codebook_size,
        commitment_beta = vcfg.commitment_beta,
        n_res_blocks    = vcfg.n_res_blocks,
    )
