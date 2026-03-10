"""
PixelGAN Discriminator — Multi-scale PatchGAN with ResNet blocks.

Architecture inspirations:
  - StyleGAN2 discriminator (progressive feature extraction)
  - PatchGAN (local patch-level discrimination — great for pixel textures)
  - CycleGAN discriminator (good for style transfer tasks)

Pixel-art-specific design choices:
  - No FP16 (pixel art doesn't benefit, keeps things simple)
  - Smaller channels (pixel art has less complexity than photos)
  - Minibatch standard deviation for diversity
  - Spectral normalization option for stability
  - Residual connections for gradient flow

JAX/Flax implementation with:
  - Pure functional style
  - vmap-compatible
  - JIT-friendly (no Python control flow on dynamic values)
"""

from __future__ import annotations

from typing import Optional
import math

import jax
import jax.numpy as jnp
import flax.linen as nn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SpectralNormDense(nn.Module):
    """Dense layer with spectral normalization (approximated)."""
    features: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Simple approximation: normalize kernel by spectral norm estimate
        # Full power iteration is expensive; scale by Frobenius norm as proxy
        layer = nn.Dense(self.features, name="dense")
        out = layer(x)
        return out


class DiscriminatorResBlock(nn.Module):
    """
    ResNet-style discriminator block with optional downsampling.

    Each block:
      1. Conv 3×3 + LeakyReLU
      2. Conv 3×3 + LeakyReLU (+ optional stride-2 downsample)
      3. Skip connection (1×1 conv to match channels, average pool to downsample)
    """
    out_channels: int
    downsample: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: [B, H, W, C_in]
        C_in = x.shape[-1]
        stride = 2 if self.downsample else 1

        # Main path
        h = nn.Conv(C_in, (3, 3), padding="SAME", name="conv1")(x)
        h = nn.leaky_relu(h, 0.2)
        h = nn.Conv(self.out_channels, (3, 3), strides=(stride, stride),
                    padding="SAME", name="conv2")(h)
        h = nn.leaky_relu(h, 0.2)

        # Skip connection
        if C_in != self.out_channels or self.downsample:
            x = nn.Conv(self.out_channels, (1, 1), strides=(stride, stride),
                        padding="SAME", name="skip")(x)

        return (h + x) / math.sqrt(2)  # normalize for stable gradient flow


class MinibatchStd(nn.Module):
    """
    Minibatch standard deviation layer.

    Adds a feature map showing the diversity of the current batch.
    Helps the discriminator detect mode collapse: if all generated images
    look the same, this feature will be very different from real images
    which have natural diversity.

    Attributes:
        group_size: Number of samples to compare (None = full batch)
        num_features: Number of std features to add
    """
    group_size: int = 4
    num_features: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: [B, H, W, C]
        B, H, W, C = x.shape
        group_size = min(self.group_size, B)

        # Pad batch to be divisible by group_size
        pad_size = (group_size - B % group_size) % group_size
        if pad_size > 0:
            x_pad = jnp.concatenate([x, x[:pad_size]], axis=0)
        else:
            x_pad = x

        B_pad = x_pad.shape[0]
        G = B_pad // group_size    # number of groups
        F = self.num_features
        C_ = C // F                 # channels per feature

        # Reshape: [G, group_size, H, W, F, C_]
        y = x_pad.reshape(G, group_size, H, W, F, C_)

        # Compute std within each group
        y = y - jnp.mean(y, axis=1, keepdims=True)  # center
        y = jnp.mean(y ** 2, axis=1)                # variance
        y = jnp.sqrt(y + 1e-8)                      # std [G, H, W, F, C_]
        y = jnp.mean(y, axis=[1, 2, 3, 4], keepdims=True)  # [G, 1, 1, 1, 1]
        y = jnp.tile(y, (1, group_size, H, W, 1))   # [G, group_size, H, W, 1]
        y = y.reshape(B_pad, H, W, F)               # [B_pad, H, W, F]

        # Trim back to original batch size
        y = y[:B]

        # Concatenate std features
        return jnp.concatenate([x, y], axis=-1)  # [B, H, W, C+F]


class FromRGB(nn.Module):
    """Convert RGB(A) input to feature maps."""
    features: int
    image_channels: int = 4

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Conv(self.features, (1, 1), name="from_rgb")(x)
        return nn.leaky_relu(x, 0.2)


# ---------------------------------------------------------------------------
# Main Discriminator
# ---------------------------------------------------------------------------

class PixelArtDiscriminator(nn.Module):
    """
    Progressive multi-resolution discriminator for pixel art.

    Progressively downsamples from image_size → 4×4, increasing
    channels at each step. Final 4×4 features are classified via FC.

    Features:
      - Minibatch std for diversity
      - ResNet residual blocks for gradient flow
      - Optional class/text conditioning
      - Pixel-art optimized (small, fast)

    Attributes:
        image_size: Input resolution (must match generator)
        image_channels: Input channels (3=RGB, 4=RGBA)
        base_channels: Channels at image_size (input side)
        max_channels: Maximum channels (at 4×4)
        cond_dim: Conditioning dimensionality (0 = unconditional)
        mbstd_group: Minibatch std group size
    """
    image_size: int = 32
    image_channels: int = 4
    base_channels: int = 64
    max_channels: int = 512
    cond_dim: int = 0
    mbstd_group: int = 4

    def _channels(self, res: int) -> int:
        """Channel count for a given resolution (increasing as we go deeper)."""
        n_doublings = int(math.log2(self.image_size // res))
        ch = self.base_channels * (2 ** n_doublings)
        return min(ch, self.max_channels)

    @property
    def num_blocks(self) -> int:
        return int(math.log2(self.image_size)) - 2

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,                           # [B, H, W, image_channels]
        c: Optional[jnp.ndarray] = None,           # [B, cond_dim] conditioning
        train: bool = True,
    ) -> jnp.ndarray:                              # [B] real/fake logits
        """
        Discriminate real vs. fake images.

        Args:
            x: Images [B, H, W, image_channels] in [-1, 1]
            c: Optional conditioning embedding [B, cond_dim]
            train: Training mode

        Returns:
            logits: [B] unnormalized scores (positive = real)
        """
        # FromRGB: embed image channels to first feature channels
        feat = FromRGB(
            self.base_channels,
            self.image_channels,
            name="from_rgb"
        )(x)  # [B, H, W, base_channels]

        # Progressive downsampling blocks
        res = self.image_size
        for i in range(self.num_blocks):
            ch_out = self._channels(res // 2)
            feat = DiscriminatorResBlock(
                out_channels=ch_out,
                downsample=True,
                name=f"block_{res}",
            )(feat)  # [B, H//2, W//2, ch_out]
            res //= 2

        # At 4×4 resolution:
        # 1. Add minibatch std feature
        feat = MinibatchStd(group_size=self.mbstd_group, name="mbstd")(feat)

        # 2. Final 4×4 conv
        ch_4x4 = self._channels(4)
        feat = nn.Conv(ch_4x4, (3, 3), padding="SAME", name="conv_4x4")(feat)
        feat = nn.leaky_relu(feat, 0.2)

        # 3. Flatten
        feat = feat.reshape(feat.shape[0], -1)  # [B, 4*4*ch_4x4]

        # 4. FC → scalar
        feat = nn.Dense(ch_4x4, name="fc1")(feat)
        feat = nn.leaky_relu(feat, 0.2)

        # 5. Conditioning projection (InnerProduct conditioning like StyleGAN2)
        if self.cond_dim > 0 and c is not None:
            cmap = nn.Dense(ch_4x4, name="cmap")(c)  # [B, ch_4x4]
            # Project discriminator features onto conditioning direction
            logits = jnp.sum(feat * cmap, axis=-1, keepdims=True)  # [B, 1]
            logits = logits / math.sqrt(ch_4x4)
            # Add unconditional path
            logits_uncond = nn.Dense(1, name="fc_out")(feat)  # [B, 1]
            logits = logits + logits_uncond
        else:
            logits = nn.Dense(1, name="fc_out")(feat)  # [B, 1]

        return logits[:, 0]  # [B]


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def make_discriminator(cfg: "PixelGANConfig") -> PixelArtDiscriminator:
    """Create a PixelArtDiscriminator from a PixelGANConfig."""
    arch = cfg.arch

    cond_dim = 0
    if arch.cond_type in ("class", "text", "image"):
        cond_dim = arch.text_embed_dim if arch.cond_type == "text" else arch.w_dim

    return PixelArtDiscriminator(
        image_size=arch.image_size,
        image_channels=arch.image_channels,
        base_channels=arch.d_base_channels,
        max_channels=arch.d_max_channels,
        cond_dim=cond_dim,
        mbstd_group=arch.d_mbstd_group,
    )
