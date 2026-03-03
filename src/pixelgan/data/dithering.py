"""
Dithering algorithms for pixel art.

Provides:
- Bayer ordered dithering (2x2, 4x4, 8x8) - classic and sharp
- Floyd-Steinberg error diffusion - smooth gradients
- Pattern dithering - classic pixel art hand-crafted patterns
- Atkinson dithering - Mac-era style, punchy look

All functions work on numpy arrays and return palette-indexed or RGB output.
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .color_palette import Color, ColorPalette


# ---------------------------------------------------------------------------
# Bayer threshold matrices
# ---------------------------------------------------------------------------

BAYER_2x2 = np.array([
    [0,  2],
    [3,  1],
], dtype=np.float32) / 4.0

BAYER_4x4 = np.array([
    [ 0,  8,  2, 10],
    [12,  4, 14,  6],
    [ 3, 11,  1,  9],
    [15,  7, 13,  5],
], dtype=np.float32) / 16.0

BAYER_8x8 = np.array([
    [ 0, 32,  8, 40,  2, 34, 10, 42],
    [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44,  4, 36, 14, 46,  6, 38],
    [60, 28, 52, 20, 62, 30, 54, 22],
    [ 3, 35, 11, 43,  1, 33,  9, 41],
    [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47,  7, 39, 13, 45,  5, 37],
    [63, 31, 55, 23, 61, 29, 53, 21],
], dtype=np.float32) / 64.0

# Classic pixel-art dithering patterns (for 2-tone transitions)
# These are hand-crafted patterns used in old games
PATTERNS_4x4 = {
    0.00: np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]),  # empty
    0.06: np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,1,0]]),  # 1/16
    0.12: np.array([[0,0,0,0],[0,0,0,0],[0,1,0,0],[0,0,0,1]]),  # 2/16
    0.25: np.array([[1,0,0,0],[0,0,1,0],[0,0,0,0],[0,1,0,0]]),  # 4/16 checkerboard
    0.50: np.array([[1,0,1,0],[0,1,0,1],[1,0,1,0],[0,1,0,1]]),  # 8/16 checkerboard
    0.75: np.array([[1,0,1,1],[1,1,1,0],[1,1,0,1],[0,1,1,1]]),  # 12/16
    1.00: np.array([[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]]),  # full
}


# ---------------------------------------------------------------------------
# Core dithering functions
# ---------------------------------------------------------------------------

def bayer_dither(
    image_float: np.ndarray,
    matrix: np.ndarray = BAYER_4x4,
    spread: float = 0.5,
) -> np.ndarray:
    """
    Apply Bayer ordered dithering to a float image.

    Args:
        image_float: [H, W] or [H, W, C] float32 in [0, 1]
        matrix: Bayer threshold matrix (tiled to image size)
        spread: Controls dither intensity (0=none, 1=full, 0.5=natural)

    Returns:
        np.ndarray: Same shape as input, values pushed toward 0 or 1
    """
    H, W = image_float.shape[:2]
    mH, mW = matrix.shape

    # Tile matrix to image size
    tiled = np.tile(matrix, (H // mH + 1, W // mW + 1))[:H, :W]
    # Shift threshold around 0.5 by spread
    threshold = 0.5 + (tiled - 0.5) * spread

    if image_float.ndim == 3:
        threshold = threshold[:, :, None]

    return (image_float >= threshold).astype(np.float32)


def floyd_steinberg(
    image_float: np.ndarray,
    palette: "ColorPalette",
) -> np.ndarray:
    """
    Floyd-Steinberg error diffusion dithering.

    Args:
        image_float: [H, W, 3] float32 in [0, 1]
        palette: ColorPalette to quantize to

    Returns:
        np.ndarray: [H, W, 3] uint8 - palette-quantized image
    """
    H, W, _ = image_float.shape
    img = (image_float * 255.0).astype(np.float32)
    result = np.zeros((H, W, 3), dtype=np.uint8)

    palette_arr = palette.to_numpy().astype(np.float32)  # [N, 3]

    for y in range(H):
        for x in range(W):
            old_pixel = img[y, x]

            # Find nearest palette color
            dists = np.sum((palette_arr - old_pixel[None]) ** 2, axis=1)
            nearest_idx = int(np.argmin(dists))
            new_pixel = palette_arr[nearest_idx]

            result[y, x] = new_pixel.astype(np.uint8)

            # Distribute quantization error (Floyd-Steinberg kernel)
            error = old_pixel - new_pixel
            if x + 1 < W:
                img[y, x + 1] += error * (7 / 16)
            if y + 1 < H:
                if x - 1 >= 0:
                    img[y + 1, x - 1] += error * (3 / 16)
                img[y + 1, x] += error * (5 / 16)
                if x + 1 < W:
                    img[y + 1, x + 1] += error * (1 / 16)

    return result


def atkinson_dither(
    image_float: np.ndarray,
    palette: "ColorPalette",
) -> np.ndarray:
    """
    Atkinson dithering - Mac classic era. Distributes only 3/4 of error,
    giving a punchy high-contrast look. Great for game sprites.

    Args:
        image_float: [H, W, 3] float32 in [0, 1]
        palette: ColorPalette to quantize to

    Returns:
        np.ndarray: [H, W, 3] uint8
    """
    H, W, _ = image_float.shape
    img = (image_float * 255.0).astype(np.float32)
    result = np.zeros((H, W, 3), dtype=np.uint8)
    palette_arr = palette.to_numpy().astype(np.float32)

    for y in range(H):
        for x in range(W):
            old_pixel = img[y, x]

            dists = np.sum((palette_arr - old_pixel[None]) ** 2, axis=1)
            nearest_idx = int(np.argmin(dists))
            new_pixel = palette_arr[nearest_idx]
            result[y, x] = new_pixel.astype(np.uint8)

            # Atkinson distributes only 6/8 = 3/4 of error
            error = (old_pixel - new_pixel) / 8.0

            # Kernel: 2 right, 1 below-left, 1 below, 1 below-right, 1 two-below
            offsets = [(0, 1), (0, 2), (1, -1), (1, 0), (1, 1), (2, 0)]
            for dy, dx in offsets:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W:
                    img[ny, nx] += error

    return result


def ordered_palette_dither(
    image_rgb: np.ndarray,
    palette: "ColorPalette",
    matrix: np.ndarray = BAYER_4x4,
    spread: float = 0.4,
) -> np.ndarray:
    """
    Ordered dithering with palette quantization.
    Fast and deterministic — great for real-time use.

    Args:
        image_rgb: [H, W, 3] uint8
        palette: ColorPalette to quantize to
        matrix: Bayer threshold matrix
        spread: Dither intensity

    Returns:
        np.ndarray: [H, W, 3] uint8
    """
    H, W = image_rgb.shape[:2]
    img_f = image_rgb.astype(np.float32) / 255.0

    # Tile matrix
    mH, mW = matrix.shape
    tiled = np.tile(matrix, (H // mH + 1, W // mW + 1))[:H, :W]  # [H, W]
    # Shift: add [-spread/2, +spread/2] noise
    noise = (tiled - 0.5) * spread  # [H, W]
    noisy = img_f + noise[:, :, None]  # [H, W, 3]
    noisy = np.clip(noisy, 0.0, 1.0)

    # Quantize each pixel to nearest palette color
    palette_arr = palette.to_numpy().astype(np.float32)  # [N, 3]
    result = np.zeros((H, W, 3), dtype=np.uint8)

    for y in range(H):
        for x in range(W):
            pixel = noisy[y, x] * 255.0
            dists = np.sum((palette_arr - pixel[None]) ** 2, axis=1)
            result[y, x] = palette_arr[np.argmin(dists)].astype(np.uint8)

    return result


def apply_dithering(
    image_rgb: np.ndarray,
    palette: "ColorPalette",
    method: str = "bayer4x4",
    intensity: float = 0.5,
) -> np.ndarray:
    """
    Apply dithering to an RGB image using the specified method.

    Args:
        image_rgb: [H, W, 3] uint8
        palette: Target ColorPalette
        method: One of 'none', 'bayer2x2', 'bayer4x4', 'bayer8x8',
                       'floyd_steinberg', 'atkinson', 'pattern'
        intensity: Dither strength 0-1

    Returns:
        np.ndarray: [H, W, 3] uint8
    """
    img_f = image_rgb.astype(np.float32) / 255.0

    if method == "none":
        # Just quantize without dithering
        palette_arr = palette.to_numpy().astype(np.float32)
        result = np.zeros_like(image_rgb)
        for y in range(image_rgb.shape[0]):
            for x in range(image_rgb.shape[1]):
                pixel = img_f[y, x] * 255.0
                dists = np.sum((palette_arr - pixel[None]) ** 2, axis=1)
                result[y, x] = palette_arr[np.argmin(dists)].astype(np.uint8)
        return result

    elif method == "bayer2x2":
        return ordered_palette_dither(image_rgb, palette, BAYER_2x2, intensity)

    elif method == "bayer4x4":
        return ordered_palette_dither(image_rgb, palette, BAYER_4x4, intensity)

    elif method == "bayer8x8":
        return ordered_palette_dither(image_rgb, palette, BAYER_8x8, intensity)

    elif method == "floyd_steinberg":
        return floyd_steinberg(img_f, palette)

    elif method == "atkinson":
        return atkinson_dither(img_f, palette)

    else:
        raise ValueError(f"Unknown dither method: {method!r}. "
                         f"Use: none, bayer2x2, bayer4x4, bayer8x8, "
                         f"floyd_steinberg, atkinson")


# ---------------------------------------------------------------------------
# Shading helpers for sprite generation
# ---------------------------------------------------------------------------

def add_dither_shading(
    base: np.ndarray,
    highlight: np.ndarray,
    shadow: np.ndarray,
    mask_highlight: np.ndarray,
    mask_shadow: np.ndarray,
    pattern: str = "bayer4x4",
    intensity: float = 0.5,
) -> np.ndarray:
    """
    Blend highlight/shadow into base sprite using dithering.
    Used to create smooth shading transitions in pixel art sprites.

    Args:
        base: [H, W, 3] uint8 - base sprite RGB
        highlight: [H, W, 3] uint8 - lighter color for highlights
        shadow: [H, W, 3] uint8 - darker color for shadows
        mask_highlight: [H, W] float32 [0,1] - where to apply highlight
        mask_shadow: [H, W] float32 [0,1] - where to apply shadow
        pattern: Dither pattern type
        intensity: Dither intensity

    Returns:
        np.ndarray: [H, W, 3] uint8
    """
    H, W = base.shape[:2]

    if pattern == "bayer4x4":
        matrix = BAYER_4x4
    elif pattern == "bayer2x2":
        matrix = BAYER_2x2
    elif pattern == "bayer8x8":
        matrix = BAYER_8x8
    else:
        matrix = BAYER_4x4

    mH, mW = matrix.shape
    tiled = np.tile(matrix, (H // mH + 1, W // mW + 1))[:H, :W]

    result = base.copy()

    # Apply highlight where mask_highlight > threshold
    h_mask = mask_highlight >= (tiled * intensity + (1 - intensity) * 0.5)
    result[h_mask] = highlight[h_mask]

    # Apply shadow where mask_shadow > threshold
    s_mask = mask_shadow >= (tiled * intensity + (1 - intensity) * 0.5)
    result[s_mask] = shadow[s_mask]

    return result


# ---------------------------------------------------------------------------
# Per-size dithering profiles
# ---------------------------------------------------------------------------
# Different sprite sizes benefit from different dithering strategies:
#   8×8   — no dithering (too noisy at this scale)
#  16×16  — light 2×2 Bayer (subtle ordered pattern)
#  32×32  — 4×4 Bayer (balanced pixel art look)
#  64×64  — Atkinson (smooth with high contrast, Mac-classic style)
# 128×128 — Floyd-Steinberg (maximum smoothness for large sprites)

SIZE_DITHER_PROFILES: dict[int, dict] = {
    8:   {"method": "none",            "intensity": 0.00},
    16:  {"method": "bayer2x2",        "intensity": 0.20},
    32:  {"method": "bayer4x4",        "intensity": 0.35},
    64:  {"method": "atkinson",        "intensity": 0.40},
    128: {"method": "floyd_steinberg", "intensity": 0.50},
}


def get_size_dither_profile(size: int) -> dict:
    """
    Return the recommended dithering method and intensity for a given image size.

    Args:
        size: Sprite size in pixels (one side of a square sprite)

    Returns:
        dict with keys "method" (str) and "intensity" (float 0-1)

    Example::

        profile = get_size_dither_profile(32)
        # {"method": "bayer4x4", "intensity": 0.35}
    """
    breakpoints = sorted(SIZE_DITHER_PROFILES.keys())
    for bp in breakpoints:
        if size <= bp:
            return SIZE_DITHER_PROFILES[bp]
    return SIZE_DITHER_PROFILES[max(breakpoints)]
