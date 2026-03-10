from .color_palette import (
    Color, ColorPalette, PaletteGenerator,
    PALETTES, PICO8, ENDESGA32, SWEETIE16, ARNE16, GAMEBOY,
    get_sprite_palette, generate_palette_variations,
    generate_tree_palette, TREE_PALETTE_CONFIGS,
    GALAGA_PALETTES, ZELDA_PALETTES, PACMAN_PALETTES,
)
from .dithering import (
    BAYER_2x2, BAYER_4x4, BAYER_8x8,
    bayer_dither, floyd_steinberg, atkinson_dither,
    ordered_palette_dither, apply_dithering,
)
from .sprite_generator import (
    SPRITES, CATEGORIES,
    SpriteRenderer,
    generate_sprite_sheet,
    generate_training_batch,
    list_sprites,
    get_sprite_info,
)
from .tree_generator import (
    ProceduralTreeGenerator,
    generate_tree_batch,
    generate_all_tree_batches,
)

# Option B: palette-indexed parquet format
try:
    from .indexed_format import (
        IndexedSprite,
        rgba_to_indexed,
        indexed_to_rgb,
        indexed_to_float,
        indexed_to_palette_array,
        save_indexed_parquet,
        load_indexed_parquet,
        is_indexed_parquet,
        decode_indexed_row,
        collect_all_colors,
        measure_color_diversity,
        build_global_palette,
        remap_to_global_palette,
    )
except ImportError:
    pass  # pyarrow not installed — indexed format unavailable
