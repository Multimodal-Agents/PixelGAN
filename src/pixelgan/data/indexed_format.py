"""
Option B: Indexed sprite format for PixelGAN datasets.

Replaces PNG-in-parquet with a palette-indexed representation:
  - index_map:  uint8 [H, W]  — palette index per pixel (0 = transparent)
  - palette:    uint8 [N, 3]  — RGB colour per index
  - n_colors:   uint8         — number of non-transparent palette entries

Benefits over PNG:
  - ~10× faster loading: no PIL decode, just np.frombuffer
  - Explicit palette: generator can directly exploit colour structure
  - Lossless and exact: every palette assignment is stored precisely
  - Transparent pixels stored as index 0 without a separate alpha mask
  - Compatible and inspectable: reconstruct RGB at any time

Parquet schema (indexed format):
  Column "seed"      int64   — reproducible generation seed
  Column "index_map" bytes   — raw uint8 bytes for [H×W] index map
  Column "palette"   bytes   — raw uint8 bytes for [N×3] RGB palette
  Column "n_colors"  int32   — number of entries in palette (excl. transparent)

Quick usage:
    from pixelgan.data.indexed_format import (
        rgba_to_indexed, indexed_to_rgb, indexed_to_float,
        save_indexed_parquet, load_indexed_parquet,
    )

    # Convert a PIL/RGBA sprite
    sprite = rgba_to_indexed(pil_image_rgba)
    rgb_arr = indexed_to_rgb(sprite)
    float_arr = indexed_to_float(sprite)       # [-1, 1] float32

    # Save/load a dataset
    save_indexed_parquet(sprites_list, seeds, "dataset.parquet")
    df = load_indexed_parquet("dataset.parquet")
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class IndexedSprite:
    """
    A palette-indexed pixel art sprite.

    index_map: uint8 [H, W]  — each pixel is an index into `palette`.
                               Index 0 is always the transparent/background
                               slot. Visible pixels use indices 1..n_colors.
    palette:   uint8 [N, 3]  — RGB colours. palette[0] is ignored (transparent).
                               palette[1..] are the sprite colours in order.
    n_colors:  int            — number of non-transparent palette entries
                               (i.e. palette has n_colors+1 rows including slot 0).
    size:      int            — sprite width/height (always square).
    """
    index_map: np.ndarray    # [H, W] uint8
    palette:   np.ndarray    # [N, 3] uint8
    n_colors:  int

    @property
    def size(self) -> int:
        return self.index_map.shape[0]

    @property
    def n_palette_entries(self) -> int:
        """Total palette rows including the transparent slot 0."""
        return self.palette.shape[0]


# ---------------------------------------------------------------------------
# Conversion: PIL RGBA → IndexedSprite
# ---------------------------------------------------------------------------

def rgba_to_indexed(
    img: Image.Image,
    max_colors: int = 16,
    bg_slot_color: tuple[int, int, int] = (0, 0, 0),
) -> IndexedSprite:
    """
    Convert a PIL RGBA image to an IndexedSprite.

    Transparent pixels (alpha == 0) are assigned to palette slot 0.
    Visible pixels are quantized to at most `max_colors` unique colours,
    assigned to slots 1..N.

    Args:
        img:             PIL image in any mode; converted to RGBA internally.
        max_colors:      Maximum number of *visible* colour slots (1–254).
        bg_slot_color:   Colour stored in slot 0 (transparent slot); never
                         rendered, just a placeholder. Default (0,0,0).

    Returns:
        IndexedSprite with palette[0] holding bg_slot_color for transparency.
    """
    img = img.convert("RGBA")
    arr = np.array(img, dtype=np.uint8)  # [H, W, 4]
    H, W = arr.shape[:2]

    alpha    = arr[:, :, 3]
    rgb      = arr[:, :, :3]

    # Collect unique visible colours
    visible_mask = alpha > 0
    visible_rgb  = rgb[visible_mask]               # [N_vis, 3]

    if len(visible_rgb) == 0:
        # Fully transparent sprite — single-entry palette, all indices = 0
        palette   = np.array([list(bg_slot_color)], dtype=np.uint8)
        idx_map   = np.zeros((H, W), dtype=np.uint8)
        return IndexedSprite(idx_map, palette, n_colors=0)

    # Find unique colours (up to max_colors)
    unique_rows = np.unique(visible_rgb.reshape(-1, 3), axis=0)
    if len(unique_rows) > max_colors:
        # Quantize using PIL's median-cut palette
        quantized = _quantize_colors(visible_rgb, max_colors)
        unique_rows = np.unique(quantized.reshape(-1, 3), axis=0)
    else:
        quantized = visible_rgb

    # Build palette: slot 0 = transparent placeholder, slots 1.. = colours
    palette = np.vstack([
        np.array([bg_slot_color], dtype=np.uint8),   # slot 0
        unique_rows.astype(np.uint8),                 # slots 1..N
    ])  # [N+1, 3]

    # Build index map
    idx_map = np.zeros((H, W), dtype=np.uint8)

    # Map each visible pixel to its palette index
    # Vectorised: build a lookup table via sorted palette rows
    # For small palettes (≤16) a simple broadcasted comparison is fastest
    rgb_flat      = rgb.reshape(-1, 3)         # [H*W, 3]
    palette_vis   = palette[1:]                # [N, 3]  colour slots only

    # Broadcast: [H*W, 1, 3] vs [1, N, 3] → [H*W, N] bool
    matches = np.all(
        rgb_flat[:, None, :] == palette_vis[None, :, :], axis=2
    )  # [H*W, N]

    # For pixels with a match, take first matching column + 1 (for slot offset)
    # Transparent pixels will not match and stay at 0
    has_match = matches.any(axis=1)
    idx_flat  = np.zeros(H * W, dtype=np.uint8)
    idx_flat[has_match] = matches[has_match].argmax(axis=1).astype(np.uint8) + 1

    idx_map = idx_flat.reshape(H, W)

    return IndexedSprite(
        index_map = idx_map,
        palette   = palette,
        n_colors  = len(palette_vis),
    )


def _quantize_colors(
    rgb: np.ndarray,
    n_colors: int,
) -> np.ndarray:
    """
    Median-cut color quantization via PIL.
    rgb: [N, 3] uint8 pixels  →  returns same shape with quantized colours.
    """
    H = len(rgb)
    tmp = Image.fromarray(rgb.reshape(1, H, 3), mode="RGB")
    quantized = tmp.quantize(colors=n_colors, method=Image.Quantize.MEDIANCUT)
    quantized_rgb = quantized.convert("RGB")
    return np.array(quantized_rgb, dtype=np.uint8).reshape(H, 3)


# ---------------------------------------------------------------------------
# Conversion: IndexedSprite → RGB / float arrays
# ---------------------------------------------------------------------------

def indexed_to_rgb(
    sprite: IndexedSprite,
    bg_rgb: tuple[int, int, int] = (40, 40, 40),
    target_size: Optional[int] = None,
) -> np.ndarray:
    """
    Render an IndexedSprite to a uint8 RGB array.

    Transparent pixels (index 0) are filled with bg_rgb.

    Args:
        sprite:      IndexedSprite to render.
        bg_rgb:      Background colour for transparent pixels.
        target_size: If given, nearest-neighbour resize to target_size×target_size.

    Returns:
        uint8 [H, W, 3] RGB array.
    """
    H, W = sprite.index_map.shape
    out = np.empty((H, W, 3), dtype=np.uint8)

    # Transparent pixels → background
    transparent = sprite.index_map == 0
    out[transparent]  = bg_rgb

    # Visible pixels → palette lookup
    vis_idx = sprite.index_map[~transparent]
    # Clamp to valid palette range (safety for malformed sprites)
    vis_idx = np.clip(vis_idx, 0, sprite.palette.shape[0] - 1)
    out[~transparent] = sprite.palette[vis_idx]

    if target_size is not None and target_size != H:
        img = Image.fromarray(out, mode="RGB")
        img = img.resize((target_size, target_size), Image.NEAREST)
        out = np.array(img, dtype=np.uint8)

    return out


def indexed_to_float(
    sprite: IndexedSprite,
    bg_rgb: tuple[int, int, int] = (40, 40, 40),
    target_size: Optional[int] = None,
) -> np.ndarray:
    """
    Render an IndexedSprite to float32 [-1, 1] RGB — the format used by
    the GAN discriminator and generator output head.

    Args:
        sprite:      IndexedSprite to render.
        bg_rgb:      Background colour for transparent pixels.
        target_size: If given, nearest-neighbour resize.

    Returns:
        float32 [H, W, 3] in [-1, 1].
    """
    rgb = indexed_to_rgb(sprite, bg_rgb=bg_rgb, target_size=target_size)
    return rgb.astype(np.float32) / 127.5 - 1.0


def indexed_to_palette_array(
    sprite: IndexedSprite,
    n_colors: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the index map and a fixed-width normalised palette suitable
    for feeding to the palette-indexed generator (Option A).

    The palette is zero-padded / truncated to exactly n_colors+1 entries
    (slot 0 = transparent, slots 1..n_colors = visible colours).
    Palette values are normalised to float32 [-1, 1] to match GAN space.

    Returns:
        idx_map:  uint8  [H, W]        — palette index per pixel
        palette:  float32 [n_colors+1, 3] — normalised palette (slot 0 = bg)
    """
    N_slots = n_colors + 1
    palette_fixed = np.zeros((N_slots, 3), dtype=np.float32)

    src = sprite.palette.astype(np.float32) / 127.5 - 1.0
    rows_to_copy = min(src.shape[0], N_slots)
    palette_fixed[:rows_to_copy] = src[:rows_to_copy]

    return sprite.index_map, palette_fixed


# ---------------------------------------------------------------------------
# Parquet I/O
# ---------------------------------------------------------------------------

def save_indexed_parquet(
    sprites: Sequence[IndexedSprite],
    seeds:   Sequence[int],
    path:    str | Path,
) -> None:
    """
    Save a list of IndexedSprites to a parquet file.

    Schema:
        seed       int64   — generation seed
        index_map  bytes   — raw uint8 bytes [H×W]
        palette    bytes   — raw uint8 bytes [N×3]
        n_colors   int32   — number of visible colour slots

    Args:
        sprites: Indexed sprites to save.
        seeds:   Matching generation seeds (one per sprite).
        path:    Output .parquet path.
    """
    import pandas as pd

    assert len(sprites) == len(seeds), \
        f"sprites ({len(sprites)}) and seeds ({len(seeds)}) must have same length"

    rows = []
    for sprite, seed in zip(sprites, seeds):
        rows.append({
            "seed":      int(seed),
            "index_map": sprite.index_map.astype(np.uint8).tobytes(),
            "palette":   sprite.palette.astype(np.uint8).tobytes(),
            "n_colors":  int(sprite.n_colors),
        })

    pd.DataFrame(rows).to_parquet(str(path), index=False)


def load_indexed_parquet(path: str | Path) -> "pd.DataFrame":
    """
    Load an indexed-format parquet file. Returns a DataFrame with columns:
        seed, index_map (bytes), palette (bytes), n_colors (int32)
    """
    import pandas as pd
    return pd.read_parquet(str(path))


def is_indexed_parquet(path: str | Path) -> bool:
    """
    Sniff the parquet schema to determine if it uses indexed format.
    Returns True if the file has 'index_map' and 'palette' columns.
    """
    import pandas as pd
    df = pd.read_parquet(str(path), columns=None)
    return "index_map" in df.columns and "palette" in df.columns


def decode_indexed_row(
    row: dict,
    target_size: int,
    bg_rgb: tuple[int, int, int] = (40, 40, 40),
    return_float: bool = True,
    n_palette_slots: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Decode one row from an indexed parquet into:
      - image: float32 [H, W, 3] RGB in [-1, 1]   (for discriminator)
      - palette: float32 [n_palette_slots+1, 3]    (for palette-indexed G head)

    Args:
        row:              Dict/Series from indexed parquet (seed, index_map, palette, n_colors).
        target_size:      Resize to this resolution if needed.
        bg_rgb:           Background colour for transparent pixels.
        return_float:     If True return float32 [-1,1]; else uint8 [0,255].
        n_palette_slots:  Fixed palette width for PaletteHead (including slot 0).

    Returns:
        (image_array, palette_array)
    """
    H = W = target_size
    n_colors = int(row["n_colors"])

    # Decode index map
    raw_idx = np.frombuffer(bytes(row["index_map"]), dtype=np.uint8)
    native_size = int(round(len(raw_idx) ** 0.5))
    idx_map = raw_idx.reshape(native_size, native_size)

    # Decode palette
    raw_pal = np.frombuffer(bytes(row["palette"]), dtype=np.uint8)
    pal_rows = raw_pal.shape[0] // 3
    palette = raw_pal.reshape(pal_rows, 3)

    sprite = IndexedSprite(
        index_map = idx_map,
        palette   = palette,
        n_colors  = n_colors,
    )

    # Render to RGB
    rgb_uint8 = indexed_to_rgb(sprite, bg_rgb=bg_rgb, target_size=target_size)

    if return_float:
        image = rgb_uint8.astype(np.float32) / 127.5 - 1.0
    else:
        image = rgb_uint8

    # Build fixed-width normalised palette
    _, palette_float = indexed_to_palette_array(sprite, n_colors=n_palette_slots - 1)

    return image, palette_float
