"""
PixelGAN Mapping Network (Z → W space).

Inspired by StyleGAN2/3's mapping network but adapted for pixel art:
- Smaller (2-8 FC layers vs StyleGAN's up to 8)
- Supports class conditioning, text conditioning, image conditioning
- W-space broadcasts to per-layer style codes

The mapping network transforms a random latent z into an intermediate
style code w that disentangles the latent space and gives much better
control over the generated images.

JAX/Flax implementation:
- Pure functional (no mutable state)
- JIT-compiled
- Explicit PRNG management
"""

from __future__ import annotations

from typing import Optional
import jax
import jax.numpy as jnp
import flax.linen as nn


class PixelNorm(nn.Module):
    """Pixelwise feature normalization — normalize latent z before mapping."""
    eps: float = 1e-8

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: [..., dim], normalize along last axis
        return x * jax.lax.rsqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + self.eps)


class MappingFC(nn.Module):
    """Single FC layer for mapping network with LR equalization support."""
    features: int
    lr_multiplier: float = 0.01  # Reduced LR for mapping net stability

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # He init scaled by lr_multiplier for equalized learning rate
        scale = 1.0 / self.lr_multiplier
        x = nn.Dense(
            self.features,
            kernel_init=nn.initializers.normal(stddev=1.0 / scale),
            bias_init=nn.initializers.zeros,
        )(x)
        return nn.leaky_relu(x, negative_slope=0.2) * self.lr_multiplier


class MappingNetwork(nn.Module):
    """
    Z → W mapping network.

    Maps a random latent vector z (and optional conditioning c) to an
    intermediate style vector w. The W-space has much better disentanglement
    properties than raw Z, enabling fine-grained control over generation.

    Attributes:
        z_dim: Input latent dimensionality
        c_dim: Conditioning label/embedding dimensionality (0 = unconditional)
        w_dim: Output style space dimensionality
        num_layers: Number of FC layers in the mapping network
        lr_multiplier: LR scaling for mapping layers (keeps them from diverging)
        w_avg_beta: EMA beta for tracking W-space mean (used for truncation)
        num_ws: Number of W-space style inputs (one per synthesis layer)
    """
    z_dim: int = 256
    c_dim: int = 0          # 0 = unconditional
    w_dim: int = 128
    num_layers: int = 4
    lr_multiplier: float = 0.01
    w_avg_beta: float = 0.998
    num_ws: int = 1          # How many W vectors to output (one per synthesis block)

    @nn.compact
    def __call__(
        self,
        z: jnp.ndarray,               # [B, z_dim]
        c: Optional[jnp.ndarray] = None,  # [B, c_dim] conditioning
        truncation_psi: float = 1.0,   # < 1 = quality/diversity tradeoff
        train: bool = True,
    ) -> jnp.ndarray:                  # [B, num_ws, w_dim]
        """
        Map latent Z to style W.

        Args:
            z: Random latent vector [B, z_dim]
            c: Optional conditioning [B, c_dim]
            truncation_psi: 1.0 = full diversity, 0.5 = higher quality/less diversity
            train: Training mode

        Returns:
            w: Style codes [B, num_ws, w_dim]
        """
        # Normalize input z (pixelwise normalization)
        x = PixelNorm()(z)

        # Embed and normalize conditioning if provided
        if self.c_dim > 0 and c is not None:
            y = nn.Dense(self.w_dim, name="embed_c")(c.astype(jnp.float32))
            # Normalize conditioning embedding too
            y = PixelNorm()(y)
            x = jnp.concatenate([x, y], axis=-1)  # [B, z_dim + w_dim]

        # FC mapping layers
        for i in range(self.num_layers):
            x = MappingFC(self.w_dim, self.lr_multiplier, name=f"fc{i}")(x)
            # x is now [B, w_dim] after first layer

        # Track W-space mean via EMA (for truncation trick)
        # Only access the variable if the collection exists or is mutable.
        if self.is_mutable_collection("ema") or self.has_variable("ema", "w_avg"):
            w_avg = self.variable(
                "ema", "w_avg",
                lambda: jnp.zeros((self.w_dim,), dtype=jnp.float32)
            )
            if train and self.is_mutable_collection("ema"):
                # Update EMA (only when collection is mutable)
                batch_mean = jnp.mean(x, axis=0)  # [w_dim]
                new_avg = w_avg.value * self.w_avg_beta + batch_mean * (1 - self.w_avg_beta)
                w_avg.value = new_avg
            w_avg_val = w_avg.value
        else:
            # Inference without saved ema state: use zeros (no truncation bias)
            w_avg_val = jnp.zeros((self.w_dim,), dtype=jnp.float32)

        # Apply truncation trick: lerp toward W mean
        if truncation_psi != 1.0:
            x = w_avg_val + (x - w_avg_val) * truncation_psi

        # Broadcast to [B, num_ws, w_dim] — one W per synthesis layer
        w = jnp.tile(x[:, None, :], (1, self.num_ws, 1))

        return w  # [B, num_ws, w_dim]


class TextEncoder(nn.Module):
    """
    Simple character/token-level text encoder for text→image conditioning.

    Much lighter than CLIP — trained from scratch with the GAN.
    Projects token sequences to a fixed-dim embedding via learned lookup + attention.

    Attributes:
        vocab_size: Token vocabulary size
        embed_dim: Token embedding dimension
        out_dim: Output conditioning vector dimension
        max_length: Maximum sequence length
    """
    vocab_size: int = 1024
    embed_dim: int = 64
    out_dim: int = 128
    max_length: int = 64

    @nn.compact
    def __call__(self, tokens: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """
        Args:
            tokens: [B, seq_len] int32 token indices
            train: Training mode

        Returns:
            [B, out_dim] conditioning vector
        """
        B, seq_len = tokens.shape

        # Learned token embeddings
        embed = nn.Embed(self.vocab_size, self.embed_dim, name="token_embed")(tokens)
        # embed: [B, seq_len, embed_dim]

        # Add positional embeddings
        pos = jnp.arange(seq_len)
        pos_embed = nn.Embed(self.max_length, self.embed_dim, name="pos_embed")(pos)
        embed = embed + pos_embed[None, :, :]  # [B, seq_len, embed_dim]

        # Simple mean-pooled attention (lightweight)
        attn_weights = nn.Dense(1, name="attn_weight")(embed)  # [B, seq_len, 1]
        attn_weights = jax.nn.softmax(attn_weights, axis=1)    # [B, seq_len, 1]
        pooled = jnp.sum(embed * attn_weights, axis=1)          # [B, embed_dim]

        # Project to output dim
        out = nn.Dense(self.out_dim, name="proj")(pooled)
        out = nn.LayerNorm(name="ln")(out)
        out = nn.gelu(out)
        out = nn.Dense(self.out_dim, name="proj2")(out)

        return out  # [B, out_dim]


class ImageEncoder(nn.Module):
    """
    Image encoder for image→image conditioning.

    Encodes a source image to a style vector that can be fed to the generator's
    mapping network. Used for image-to-image translation mode.

    Architecture: Small ResNet-style encoder, progressively downsamples.

    Attributes:
        image_size: Input image size
        image_channels: Number of input image channels
        out_dim: Output style vector dimension
        base_channels: Starting channel count
    """
    image_size: int = 32
    image_channels: int = 4
    out_dim: int = 256
    base_channels: int = 32

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """
        Args:
            x: [B, H, W, C] input image, float32 in [-1, 1]
            train: Training mode

        Returns:
            [B, out_dim] style encoding
        """
        ch = self.base_channels

        # Initial conv
        x = nn.Conv(ch, (3, 3), padding="SAME", name="conv_in")(x)
        x = nn.leaky_relu(x, 0.2)

        # Downsample blocks
        res = self.image_size
        while res > 4:
            ch_out = min(ch * 2, 512)
            # ResNet block
            residual = nn.Conv(ch_out, (1, 1), strides=(2, 2), name=f"skip_{res}")(x)
            x = nn.Conv(ch, (3, 3), padding="SAME", name=f"conv1_{res}")(x)
            x = nn.leaky_relu(x, 0.2)
            x = nn.Conv(ch_out, (3, 3), strides=(2, 2), padding="SAME", name=f"conv2_{res}")(x)
            x = x + residual
            x = nn.leaky_relu(x, 0.2)
            ch = ch_out
            res //= 2

        # Flatten and project
        x = x.reshape(x.shape[0], -1)  # [B, 4*4*ch]
        x = nn.Dense(self.out_dim, name="proj")(x)
        x = nn.LayerNorm(name="ln")(x)
        x = nn.gelu(x)
        x = nn.Dense(self.out_dim, name="proj2")(x)

        return x  # [B, out_dim]
