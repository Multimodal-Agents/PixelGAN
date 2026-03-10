"""
ZzSprite procedural sprite generator — Python port.

Original ZzSprite.js by Frank Force (https://github.com/KilledByAPixel/ZzSprite)
License: MIT

ZzSprite generates small pixel art sprites using a simple xor-shift PRNG.
Each sprite is symmetric left-right, built from an elliptical density mask,
with optional outline and four colour modes.

Key design:
  - XOR-shift32 PRNG matches the JavaScript source exactly (verified)
  - Colour and shape use independent PRNG branches (colorSeed offset trick)
  - Output: RGBA PIL Image with genuine transparency for non-sprite pixels

Usage:
    from pixelgan.data.zzsprite_generator import ZzSpriteGenerator

    gen = ZzSpriteGenerator()
    img = gen.generate(seed=42, size=16, mode=0)  # PIL Image RGBA
    img.save("sprite.png")

    # Training batch
    batch = gen.generate_batch(n=200, size=32, base_seed=0)
    # returns list of {"image_bytes": bytes, "seed": int, "caption": str}
"""

from __future__ import annotations

import io
from typing import Optional

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# XOR-shift PRNG (32-bit signed, matches JavaScript exactly)
# ---------------------------------------------------------------------------

def _to_int32(n: int) -> int:
    """Wrap to 32-bit signed integer (mirrors JS bitwise behaviour)."""
    n = n & 0xFFFFFFFF
    if n >= 0x80000000:
        n -= 0x100000000
    return n


def _xorshift(seed: int) -> int:
    """One XOR-shift step (32-bit signed, identical to ZzSprite.js Random helper)."""
    seed = _to_int32(seed ^ (seed << 13))
    seed = _to_int32(seed ^ ((seed & 0xFFFFFFFF) >> 17))  # logical (unsigned) right shift
    seed = _to_int32(seed ^ (seed << 5))
    return seed


def _rand(seed: int, max_val: float = 1.0, min_val: float = 0.0):
    """Advance PRNG and return (new_seed, value) where value ∈ [min_val, max_val)."""
    seed = _xorshift(seed)
    value = (abs(seed) % 1_000_000_000) / 1e9 * (max_val - min_val) + min_val
    return seed, value


# ---------------------------------------------------------------------------
# HSL → RGB
# ---------------------------------------------------------------------------

def _hsl_to_rgb(h: float, s_pct: float, l_pct: float) -> tuple:
    """
    Convert HSL to uint8 RGB.
      h:     0-360 degrees
      s_pct: 0-200 (matches ZzSprite.js; values >100 are clamped to 100%)
      l_pct: 0-100
    """
    h_norm = (h % 360.0) / 360.0
    s_norm = min(s_pct / 100.0, 1.0)
    l_norm = l_pct / 100.0

    if s_norm == 0.0:
        v = int(round(l_norm * 255))
        return v, v, v

    def hue2rgb(p: float, q: float, t: float) -> float:
        t = t % 1.0
        if t < 1 / 6:
            return p + (q - p) * 6 * t
        if t < 1 / 2:
            return q
        if t < 2 / 3:
            return p + (q - p) * (2 / 3 - t) * 6
        return p

    q = l_norm * (1 + s_norm) if l_norm < 0.5 else l_norm + s_norm - l_norm * s_norm
    p = 2 * l_norm - q
    r = hue2rgb(p, q, h_norm + 1 / 3)
    g = hue2rgb(p, q, h_norm)
    b = hue2rgb(p, q, h_norm - 1 / 3)
    return int(round(r * 255)), int(round(g * 255)), int(round(b * 255))


# ---------------------------------------------------------------------------
# Mode constants
# ---------------------------------------------------------------------------

MODE_COLORED    = 0  # random HSL colours with black outline
MODE_GRAYSCALE  = 1  # black / gray / white tones
MODE_SILHOUETTE = 2  # solid white fill, black outline
MODE_BLACK      = 3  # three-pass solid black (stencil / icon)

MODE_NAMES = {
    MODE_COLORED:    "colored",
    MODE_GRAYSCALE:  "grayscale",
    MODE_SILHOUETTE: "silhouette",
    MODE_BLACK:      "black",
}


# ---------------------------------------------------------------------------
# Core sprite renderer
# ---------------------------------------------------------------------------

def _generate_canvas(
    size: int,
    seed: int,
    mode: int = MODE_COLORED,
    mutate_seed: int = 0,
    color_seed: int = 0,
) -> np.ndarray:
    """
    Generate a ZzSprite pixel art sprite as a uint8 RGBA numpy array.

    Args:
        size:         Canvas side length in pixels.
        seed:         Shape seed (determines silhouette).
        mode:         Colour mode (0-3).
        mutate_seed:  Mutation offset (alters proportions within same seed family).
        color_seed:   Colour offset (alters palette without changing silhouette).

    Returns:
        uint8 [size, size, 4] RGBA array. Non-sprite pixels have alpha = 0.
    """
    canvas = np.zeros((size, size, 4), dtype=np.uint8)

    # ── Phase 1: shape parameters (mirrors ZzSprite.js setup) ──────────────
    rs = seed

    rs, v      = _rand(rs)
    flip_axis  = v < 0.5

    w = (size - 3)      if flip_axis else (size // 2 - 1)   # x half-width grid
    h = (size - 3)      if not flip_axis else (size // 2 - 1)  # y height grid

    # Jump to mutation region: JS does `randomSeed += mutateSeed + 1e8`
    ms = _to_int32(rs + mutate_seed + 100_000_000)
    ms, sprite_size = _rand(ms, 0.9, 0.6)
    sprite_size *= size
    ms, density     = _rand(ms, 1.0, 0.9)
    ms, v           = _rand(ms)
    double_center   = 1 if v < 0.5 else 0
    ms, y_bias      = _rand(ms, 0.1, -0.1)

    color_rand = 0.08 if mode == MODE_GRAYSCALE else 0.04

    # Canvas centre (JS: x += size/2|0;  y += 2)
    cx = size // 2
    cy = 2

    # ── Phase 2: draw function ──────────────────────────────────────────────
    def draw_sprite_internal(outline: bool) -> None:
        draw_seed     = seed   # reset shape PRNG each call (mirrors JS `randomSeed = seed`)
        current_color = (0, 0, 0, 255)

        pass_count = 3 if mode == MODE_BLACK else 1

        for _pass in range(pass_count):
            for k in range(w * h):
                # Grid coordinates (flip_axis transposes i/j)
                if flip_axis:
                    i = k // w   # x distance from centre
                    j = k % w    # y row
                else:
                    i = k % w    # x distance from centre
                    j = k // w   # y row

                # ── Colour branch: independent PRNG via seed offset ──────
                cs = _to_int32(draw_seed + color_seed + 1_000_000_000)
                cs, h_val = _rand(cs, 360.0, 0.0)
                h_int = int(h_val)

                if outline or mode == MODE_BLACK:
                    new_color = (0, 0, 0, 255)
                elif mode == MODE_GRAYSCALE:
                    r = h_int % 3
                    new_color = (255, 255, 255, 255) if r == 0 \
                        else (68, 68, 68, 255) if r == 1 \
                        else (153, 153, 153, 255)
                elif mode == MODE_SILHOUETTE:
                    new_color = (255, 255, 255, 255)
                else:  # MODE_COLORED
                    cs, s_val = _rand(cs, 200.0, 0.0)
                    cs, l_val = _rand(cs, 100.0, 20.0)
                    rgb = _hsl_to_rgb(h_int, s_val, l_val)
                    new_color = (rgb[0], rgb[1], rgb[2], 255)

                # Change colour with probability color_rand (first pixel always changes)
                cs, change_v = _rand(cs)
                if k == 0 or change_v < color_rand:
                    current_color = new_color

                # ── Shape branch: main sequential PRNG ───────────────────
                draw_seed, hole_v = _rand(draw_seed)
                is_hole = hole_v > density

                draw_seed, dist_v = _rand(draw_seed, sprite_size / 2)

                target_y = (1.0 - 2.0 * y_bias) * h / 2.0
                in_shape  = dist_v ** 2 > i * i + (j - target_y) ** 2

                if in_shape and not is_hole:
                    o         = 1 if outline else 0
                    draw_size = 1 + 2 * o

                    # Mirror: draw at right half (cx+i) and left half (cx-i)
                    for base_x, base_y in (
                        (cx + i - o - double_center, cy + j - o),
                        (cx - i - o,                 cy + j - o),
                    ):
                        for dy in range(draw_size):
                            for dx in range(draw_size):
                                px = base_x + dx
                                py = base_y + dy
                                if 0 <= px < size and 0 <= py < size:
                                    canvas[py, px] = current_color

    # ── Phase 3: outline pass then fill pass ────────────────────────────────
    if mode != MODE_BLACK:
        draw_sprite_internal(outline=True)
    draw_sprite_internal(outline=False)

    return canvas


# ---------------------------------------------------------------------------
# Public generator class
# ---------------------------------------------------------------------------

class ZzSpriteGenerator:
    """
    Procedural pixel art sprite generator based on ZzSprite.js (Frank Force).

    Generates organic-looking sprites with guaranteed symmetry.
    Each unique (seed, mutate_seed, color_seed, mode) combination → one sprite.

    Naturally produces:
      - Creatures / enemies / power-ups  (MODE_COLORED)
      - Rock / metal / bone items        (MODE_GRAYSCALE)
      - Icon silhouettes                 (MODE_SILHOUETTE)
      - Stencil / UI icons               (MODE_BLACK)

    Output is RGBA with genuine transparency for non-sprite pixels.
    Compatible with create_seed_dataset() and create_text_dataset() via
    generate_batch() which returns the same dict format as tree_generator.py.
    """

    MODE_LABELS = {
        MODE_COLORED:    "colorful",
        MODE_GRAYSCALE:  "grayscale",
        MODE_SILHOUETTE: "silhouette",
        MODE_BLACK:      "stencil",
    }

    def generate(
        self,
        seed: int = 1,
        size: int = 16,
        mode: int = MODE_COLORED,
        mutate_seed: int = 0,
        color_seed: int = 0,
    ) -> Image.Image:
        """
        Generate one ZzSprite as a PIL RGBA Image.

        Args:
            seed:         Shape seed.
            size:         Canvas size (sprite fits within size × size).
            mode:         0=colored, 1=grayscale, 2=silhouette, 3=black stencil.
            mutate_seed:  Mutation offset (same shape, different proportions).
            color_seed:   Colour offset (same shape, different colours).

        Returns:
            PIL Image in RGBA mode.
        """
        canvas = _generate_canvas(size, seed, mode, mutate_seed, color_seed)
        return Image.fromarray(canvas, mode="RGBA")

    def generate_batch(
        self,
        n: int,
        size: int = 16,
        base_seed: int = 0,
        modes: Optional[list] = None,
        mutate_variants: int = 3,
        color_variants: int = 2,
        include_flipped: bool = True,
    ) -> list:
        """
        Generate a batch of ZzSprite sprites for dataset creation.

        Args:
            n:               Target number of sprites.
            size:            Canvas size.
            base_seed:       Starting seed (incremented per unique shape).
            modes:           Modes to cycle through. Default: all 4.
            mutate_variants: Mutation variants per seed.
            color_variants:  Colour variants per seed × mutation.
            include_flipped: Include horizontally flipped copies (doubles dataset).

        Returns:
            List of dicts: {"image_bytes": bytes, "seed": int, "caption": str}
        """
        if modes is None:
            modes = [MODE_COLORED, MODE_GRAYSCALE, MODE_SILHOUETTE, MODE_BLACK]

        samples = []
        seed = base_seed

        while len(samples) < n:
            for mode in modes:
                for m_var in range(mutate_variants):
                    for c_var in range(color_variants):
                        if len(samples) >= n:
                            break

                        img = self.generate(
                            seed=seed, size=size, mode=mode,
                            mutate_seed=m_var * 7,
                            color_seed=c_var * 13,
                        )
                        caption = self._make_caption(size, mode, m_var)
                        samples.append({
                            "image_bytes": _to_png_bytes(img),
                            "seed": seed,
                            "caption": caption,
                        })

                        if include_flipped and len(samples) < n:
                            flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
                            samples.append({
                                "image_bytes": _to_png_bytes(flipped),
                                "seed": seed + 1_000_000,
                                "caption": caption + " mirrored",
                            })

                    if len(samples) >= n:
                        break
                if len(samples) >= n:
                    break
            seed += 1

        return samples[:n]

    def _make_caption(self, size: int, mode: int, mutate_idx: int) -> str:
        mode_label  = self.MODE_LABELS.get(mode, "sprite")
        variant     = ["compact", "normal", "elongated"][min(mutate_idx, 2)]
        return f"zzsprite {mode_label} {size}px {variant} pixel art creature"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_png_bytes(img: Image.Image) -> bytes:
    """Encode PIL Image as PNG bytes."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def is_blank(img: Image.Image, min_opaque_fraction: float = 0.05) -> bool:
    """
    Return True if the sprite is essentially empty (fewer than 5% opaque pixels).
    Use to filter degenerate outputs before adding to a training dataset.
    """
    arr   = np.array(img.convert("RGBA"))
    total = arr.shape[0] * arr.shape[1]
    opaque = int(np.sum(arr[:, :, 3] > 0))
    return opaque / max(total, 1) < min_opaque_fraction
