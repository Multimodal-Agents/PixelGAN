# Dataset utilities (no JAX required)
from .dataset import (
    SeedDataset, TextDataset, ImagePairDataset,
    load_dataset, infinite_loader,
    create_seed_dataset, create_text_dataset, create_image_pair_dataset,
    decode_image, encode_image, tokenize_text,
)

# JAX-dependent (losses, trainer) — only import when JAX is available
try:
    from .losses import (
        generator_loss, discriminator_loss,
        r1_gradient_penalty, path_length_regularization,
        cycle_consistency_loss, reconstruction_loss,
        palette_coherence_loss, total_variation_loss,
        compute_g_loss, compute_d_loss,
    )
    from .trainer import PixelGANTrainer
    # Option C: VQ-VAE Stage 1 trainer
    from .vqvae_trainer import VQVAETrainer, load_vqvae_decoder
except ImportError:
    pass
