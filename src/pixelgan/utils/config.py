"""
PixelGAN configuration management.

All hyperparameters in one place, with presets tuned for each pixel art size.
Sizes use the "bit" naming convention:
  8  → 8×8   (NES icon tiles)
  32 → 32×32 (SNES/16-bit sprites)
  64 → 64×64 (N64 era)
  128→ 128×128
  256→ 256×256

Performance targets vs StyleGAN3 (256×256 reference):
  StyleGAN3 params: ~30M   | PixelGAN 256: ~1.2M  (25× smaller)
  StyleGAN3 inf:   ~100ms  | PixelGAN 256: ~1ms   (100× faster, JIT)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Resolution-specific architecture configs
# ---------------------------------------------------------------------------

@dataclass
class ArchConfig:
    """Generator + Discriminator architecture config."""

    # --- Resolution ---
    image_size: int = 32          # Pixel art native size (8/32/64/128/256)
    image_channels: int = 3       # 3=RGB (composite on bg), 4=RGBA

    # --- Latent space ---
    z_dim: int = 256              # Input noise dimensionality
    w_dim: int = 128              # Intermediate style space dimensionality
    w_num_layers: int = 4         # Mapping network depth (FC layers)
    w_lr_multiplier: float = 0.01 # LR multiplier for mapping net (stability)

    # --- Generator ---
    g_base_channels: int = 256    # Channels at 4×4 (start resolution)
    g_min_channels: int = 16      # Minimum channels at max resolution
    noise_injection: bool = True  # Per-layer stochastic noise

    # --- Discriminator ---
    d_base_channels: int = 128    # Channels at max resolution (input side)
    d_max_channels: int = 512     # Maximum channels at lowest resolution
    d_mbstd_group: int = 4        # Minibatch std group size (diversity)
    d_num_layers: int = 3         # PatchGAN extra conv layers

    # --- Output mode ---
    output_mode: str = "rgb"      # "rgb" | "palette_indexed"
    n_palette_colors: int = 8     # Palette size for palette_indexed mode
    palette_bg_rgb: Tuple[int, int, int] = (40, 40, 40)  # Transparent → this BG

    # --- Conditioning ---
    cond_type: str = "none"       # "none", "class", "text", "image"
    num_classes: int = 0          # For class conditioning
    text_embed_dim: int = 128     # Text embedding dimension
    text_vocab_size: int = 1024   # Simple token vocabulary size
    text_max_length: int = 64     # Max text token length

    # --- Computed ---
    @property
    def g_channels_per_res(self) -> dict[int, int]:
        """Channel count for each resolution in the generator."""
        channels = {}
        res = 4
        ch = self.g_base_channels
        while res <= self.image_size:
            channels[res] = max(ch, self.g_min_channels)
            ch = ch // 2
            res *= 2
        return channels

    @property
    def n_synthesis_blocks(self) -> int:
        """Number of upsampling blocks needed."""
        import math
        return int(math.log2(self.image_size)) - 2  # 4→size requires log2(size)-2 upsamples

    @property
    def d_channels_per_res(self) -> dict[int, int]:
        """Channel count for each resolution in the discriminator."""
        channels = {}
        res = self.image_size
        ch = self.d_base_channels
        while res >= 4:
            channels[res] = min(ch, self.d_max_channels)
            ch = ch * 2
            res //= 2
        return channels


@dataclass
class VQVAEConfig:
    """
    Configuration for the VQ-VAE (Option C).

    The VQ-VAE is trained in Stage 1 to compress 64×64 pixel art images
    into an 8×8 discrete latent space.  The GAN (Stage 2) then generates
    directly in that compact latent space, with the frozen VQ-VAE decoder
    expanding it back to 64×64.

    Why these defaults:
      codebook_size=256  — enough codes for pixel art diversity (vs 8192 for photos)
      latent_dim=64      — compact enough to train fast; rich enough for pixel art
      base_channels=64   — encoder/decoder channel budget (≈180k+280k params)
      n_res_blocks=2     — two residual blocks per scale; sufficient for 64px art
      commitment_beta=0.25 — standard value from VQ-VAE-2 paper
    """
    codebook_size:    int   = 256     # K: number of discrete codes
    latent_dim:       int   = 64      # D: code vector dimension
    base_channels:    int   = 64      # encoder/decoder channel base
    n_res_blocks:     int   = 2       # residual blocks per scale level
    commitment_beta:  float = 0.25    # weight of commitment loss term
    ema_decay:        float = 0.99    # EMA codebook update decay

    # Training (Stage 1)
    lr:               float = 1e-3    # Adam LR for VQ-VAE
    batch_size:       int   = 16      # VQ-VAE training batch size
    total_steps:      int   = 10_000  # Stage 1 training steps
    snapshot_steps:   int   = 1_000   # Save checkpoint every N steps
    lambda_recon:     float = 1.0     # Reconstruction loss weight
    lambda_vq:        float = 1.0     # VQ loss weight

    # Saved weights path (filled in by training script)
    checkpoint_path:  str   = "runs/vqvae/checkpoint"


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    # --- Optimizers ---
    g_lr: float = 2e-4            # Generator learning rate
    d_lr: float = 2e-4            # Discriminator learning rate
    g_beta1: float = 0.0          # Generator Adam beta1 (0 = StyleGAN style)
    g_beta2: float = 0.99         # Generator Adam beta2
    d_beta1: float = 0.0          # Discriminator Adam beta1
    d_beta2: float = 0.99         # Discriminator Adam beta2

    # --- Batch ---
    batch_size: int = 16          # Training batch size
    grad_accumulate: int = 1      # Gradient accumulation steps

    # --- Regularization ---
    r1_gamma: float = 10.0        # R1 gradient penalty weight
    r1_interval: int = 16         # Apply R1 every N discriminator steps
    pl_weight: float = 2.0        # Path length regularization weight
    pl_interval: int = 4          # Apply PL reg every N generator steps
    pl_decay: float = 0.01        # Path length EMA decay
    style_mixing_prob: float = 0.9 # Probability of style mixing during training

    # --- EMA ---
    ema_kimg: float = 10.0        # Generator EMA half-life (thousands of imgs)
    ema_rampup: float = 0.05      # EMA ramp-up coefficient

    # --- ADA (Adaptive Data Augmentation) ---
    ada_target: float = 0.6       # Target discriminator sign for real images
    ada_interval: int = 4         # ADA adjustment frequency
    ada_kimg: float = 500.0       # ADA adjustment speed

    # --- Training duration ---
    total_kimg: int = 5000        # Training length in thousands of images
    snapshot_kimg: int = 50       # Save checkpoint every N kimg
    sample_kimg: int = 5          # Save samples every N kimg

    # --- Data ---
    dataset_type: str = "seed"    # "seed", "text", "image_pair"
    num_workers: int = 4

    # --- Misc ---
    seed: int = 42
    resume_from: Optional[str] = None
    output_dir: str = "runs/pixelgan"
    mixed_precision: bool = False  # FP16 (requires CUDA)


@dataclass
class PixelGANConfig:
    """Complete PixelGAN configuration."""
    arch:     ArchConfig     = field(default_factory=ArchConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    vqvae:    Optional[VQVAEConfig] = None  # Set to enable VQ-VAE (Option C)
    name:     str = "pixelgan"


# ---------------------------------------------------------------------------
# Size-tuned presets
# ---------------------------------------------------------------------------

def make_config_8bit() -> PixelGANConfig:
    """8×8 sprite model — tiny, blazing fast, NES icon style."""
    return PixelGANConfig(
        name="pixelgan_8bit",
        arch=ArchConfig(
            image_size=8,
            image_channels=3,
            z_dim=128,
            w_dim=64,
            w_num_layers=2,
            g_base_channels=128,
            g_min_channels=32,
            d_base_channels=64,
            d_max_channels=256,
            d_mbstd_group=4,
        ),
        training=TrainingConfig(
            batch_size=64,
            g_lr=2e-4,
            d_lr=2e-4,
            r1_gamma=5.0,
            total_kimg=1000,
        ),
    )


def make_config_32bit() -> PixelGANConfig:
    """32×32 sprite model — SNES/16-bit era quality."""
    return PixelGANConfig(
        name="pixelgan_32bit",
        arch=ArchConfig(
            image_size=32,
            image_channels=3,
            z_dim=256,
            w_dim=128,
            w_num_layers=4,
            g_base_channels=256,
            g_min_channels=16,
            d_base_channels=64,
            d_max_channels=512,
        ),
        training=TrainingConfig(
            batch_size=32,
            g_lr=2e-4,
            d_lr=2e-4,
            r1_gamma=10.0,
            total_kimg=3000,
        ),
    )


def make_config_16bit() -> PixelGANConfig:
    """16×16 sprite model — retro icon scale, fast training."""
    return PixelGANConfig(
        name="pixelgan_16bit",
        arch=ArchConfig(
            image_size=16,
            image_channels=3,
            z_dim=128,
            w_dim=64,
            w_num_layers=2,
            g_base_channels=128,
            g_min_channels=16,
            d_base_channels=64,
            d_max_channels=256,
            d_mbstd_group=4,
        ),
        training=TrainingConfig(
            batch_size=64,
            g_lr=2e-4,
            d_lr=2e-4,
            r1_gamma=5.0,
            total_kimg=2000,
        ),
    )


def make_config_64bit() -> PixelGANConfig:
    """
    64×64 sprite model — N64 era, detailed sprites.

    Channel sizes are intentionally conservative vs photo-realistic GANs:
    pixel art has very limited palette + hard edges, so large channel counts
    waste memory without quality gains. Tuned to fit in 8 GB VRAM at bs=4
    with grad_accumulate=2 (effective batch = 8).
    """
    return PixelGANConfig(
        name="pixelgan_64bit",
        arch=ArchConfig(
            image_size=64,
            image_channels=3,
            z_dim=256,
            w_dim=128,
            w_num_layers=4,
            g_base_channels=128,   # was 512 — pixel art doesn't need 512 base ch
            g_min_channels=16,
            d_base_channels=32,    # reduced 64→32: prevents D memorizing 2340-sample dataset
            d_max_channels=128,    # reduced 256→128
        ),
        training=TrainingConfig(
            batch_size=4,          # was 16 — reduces activation memory 4×
            grad_accumulate=1,
            g_lr=2e-4,
            d_lr=1e-4,             # half of g_lr: D trains slower so G can keep up
            r1_gamma=10.0,         # effective per step = gamma after lazy-reg interval scaling
            total_kimg=5000,
        ),
    )


def make_config_128bit() -> PixelGANConfig:
    """128×128 sprite model — high detail pixel art."""
    return PixelGANConfig(
        name="pixelgan_128bit",
        arch=ArchConfig(
            image_size=128,
            image_channels=3,
            z_dim=512,
            w_dim=256,
            w_num_layers=6,
            g_base_channels=512,
            g_min_channels=32,
            d_base_channels=64,
            d_max_channels=512,
        ),
        training=TrainingConfig(
            batch_size=8,
            g_lr=1e-4,
            d_lr=1e-4,
            r1_gamma=20.0,
            total_kimg=10000,
        ),
    )


def make_config_256bit() -> PixelGANConfig:
    """256×256 sprite model — ultra-detailed pixel art scenes/characters."""
    return PixelGANConfig(
        name="pixelgan_256bit",
        arch=ArchConfig(
            image_size=256,
            image_channels=3,
            z_dim=512,
            w_dim=256,
            w_num_layers=8,
            g_base_channels=512,
            g_min_channels=32,
            d_base_channels=64,
            d_max_channels=512,
        ),
        training=TrainingConfig(
            batch_size=4,
            g_lr=1e-4,
            d_lr=1e-4,
            r1_gamma=40.0,
            total_kimg=25000,
        ),
    )


SIZE_PRESETS: dict[int, PixelGANConfig] = {
    8:   make_config_8bit(),
    16:  make_config_16bit(),
    32:  make_config_32bit(),
    64:  make_config_64bit(),
    128: make_config_128bit(),
    256: make_config_256bit(),
}

VALID_SIZES = [8, 16, 32, 64, 128, 256]


def get_config(size: int, **overrides) -> PixelGANConfig:
    """
    Get config for a given pixel art size with optional overrides.

    Args:
        size: Pixel art image size (8/32/64/128/256)
        **overrides: Override specific fields, e.g. batch_size=64, g_lr=1e-3

    Returns:
        PixelGANConfig

    Example:
        cfg = get_config(32, batch_size=64, total_kimg=1000)
    """
    if size not in SIZE_PRESETS:
        raise ValueError(f"size must be one of {VALID_SIZES}, got {size}")

    cfg = SIZE_PRESETS[size]

    # Apply overrides
    for k, v in overrides.items():
        if hasattr(cfg.arch, k):
            setattr(cfg.arch, k, v)
        elif hasattr(cfg.training, k):
            setattr(cfg.training, k, v)
        else:
            raise KeyError(f"Unknown config key: {k!r}")

    return cfg

