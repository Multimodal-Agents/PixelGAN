"""
PixelGAN — Hyper-optimized GAN for pixel art generation.

A JAX/Flax-based generative adversarial network designed specifically for
pixel art. Inspired by StyleGAN3 but purpose-built for small, crisp sprites.

Key features:
  - ~25x smaller than StyleGAN3 for same resolution
  - ~100x faster inference (JIT-compiled, tiny model)
  - Three training modes: seed→image, text→image, image→image
  - Parquet dataset support (2 columns per dataset type)
  - Pixel-perfect upsampling (pixel shuffle, no bilinear blurring)
  - Style-modulated synthesis (W-space like StyleGAN2)
  - Built-in sprite generator with Galaga/Zelda/Pacman sprites
  - Professional color palette system
  - Multiple dithering modes
"""

__version__ = "0.1.0"
__author__ = "PixelGAN Team"

# Always available (no JAX required)
from .utils.config import get_config, PixelGANConfig, VQVAEConfig, VALID_SIZES
from .data.sprite_generator import (
    SPRITES, CATEGORIES,
    generate_training_batch,
    generate_sprite_sheet,
    list_sprites,
)
from .data.color_palette import (
    PaletteGenerator, ColorPalette,
    PALETTES, get_sprite_palette,
)

# JAX-dependent imports — only available when JAX is installed
try:
    from .models.generator import make_generator, PixelArtGenerator
    from .models.discriminator import make_discriminator, PixelArtDiscriminator
    from .models.palette_head import (
        ToPaletteLogits, PaletteLookup,
        get_palette_temperature, decode_to_indices,
    )
    from .models.vqvae import VQVAE, make_vqvae, vqvae_loss
    from .training.trainer import PixelGANTrainer
    from .training.vqvae_trainer import VQVAETrainer, load_vqvae_decoder
    from .training.dataset import (
        load_dataset, SeedDataset, TextDataset, ImagePairDataset,
    )
    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False

__all__ = [
    # Config (always available)
    "get_config", "PixelGANConfig", "VQVAEConfig", "VALID_SIZES",
    # Data generation (always available)
    "SPRITES", "CATEGORIES",
    "generate_training_batch", "generate_sprite_sheet", "list_sprites",
    "PaletteGenerator", "ColorPalette", "PALETTES", "get_sprite_palette",
]

if _JAX_AVAILABLE:
    __all__ += [
        "make_generator", "PixelArtGenerator",
        "make_discriminator", "PixelArtDiscriminator",
        "ToPaletteLogits", "PaletteLookup",
        "get_palette_temperature", "decode_to_indices",
        "VQVAE", "make_vqvae", "vqvae_loss",
        "PixelGANTrainer", "VQVAETrainer", "load_vqvae_decoder",
        "load_dataset", "SeedDataset", "TextDataset", "ImagePairDataset",
    ]
