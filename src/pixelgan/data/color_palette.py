"""
Color palette system for pixel art generation.

Provides:
- Named classic palettes (NES, PICO-8, Game Boy, Endesga32, etc.)
- Procedural palette generation with color theory (harmonic, triadic, etc.)
- Palette quantization for dithering output
- Color distance metrics (perceptual, Euclidean)

The palette system is the aesthetic heart of the sprite generator.
"""

from __future__ import annotations

import colorsys
import math
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------

@dataclass
class Color:
    r: int  # 0-255
    g: int  # 0-255
    b: int  # 0-255
    a: int = 255  # 0-255, alpha

    def to_tuple(self) -> tuple[int, int, int, int]:
        return (self.r, self.g, self.b, self.a)

    def to_rgb_tuple(self) -> tuple[int, int, int]:
        return (self.r, self.g, self.b)

    def to_hex(self) -> str:
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}"

    def to_float(self) -> tuple[float, float, float]:
        return (self.r / 255.0, self.g / 255.0, self.b / 255.0)

    @classmethod
    def from_hex(cls, hex_str: str) -> "Color":
        hex_str = hex_str.lstrip("#")
        return cls(
            r=int(hex_str[0:2], 16),
            g=int(hex_str[2:4], 16),
            b=int(hex_str[4:6], 16),
        )

    @classmethod
    def from_hsl(cls, h: float, s: float, l: float) -> "Color":
        """h: 0-360, s: 0-1, l: 0-1"""
        r, g, b = colorsys.hls_to_rgb(h / 360.0, l, s)
        return cls(int(r * 255), int(g * 255), int(b * 255))

    def to_hsl(self) -> tuple[float, float, float]:
        h, l, s = colorsys.rgb_to_hls(self.r / 255.0, self.g / 255.0, self.b / 255.0)
        return (h * 360.0, s, l)

    def perceptual_distance(self, other: "Color") -> float:
        """CIE76 perceptual color distance."""
        dr = (self.r - other.r) * 0.299
        dg = (self.g - other.g) * 0.587
        db = (self.b - other.b) * 0.114
        return math.sqrt(dr * dr + dg * dg + db * db)

    def lighten(self, amount: float) -> "Color":
        h, s, l = self.to_hsl()
        l = min(1.0, l + amount)
        return Color.from_hsl(h, s, l)

    def darken(self, amount: float) -> "Color":
        h, s, l = self.to_hsl()
        l = max(0.0, l - amount)
        return Color.from_hsl(h, s, l)

    def saturate(self, amount: float) -> "Color":
        h, s, l = self.to_hsl()
        s = min(1.0, s + amount)
        return Color.from_hsl(h, s, l)

    def shift_hue(self, degrees: float) -> "Color":
        h, s, l = self.to_hsl()
        return Color.from_hsl((h + degrees) % 360, s, l)

    def with_alpha(self, a: int) -> "Color":
        return Color(self.r, self.g, self.b, a)

    TRANSPARENT = None  # Initialized below

Color.TRANSPARENT = Color(0, 0, 0, 0)


@dataclass
class ColorPalette:
    """A named set of colors for pixel art."""
    name: str
    colors: list[Color]
    description: str = ""

    def __len__(self) -> int:
        return len(self.colors)

    def __getitem__(self, idx: int) -> Color:
        return self.colors[idx]

    def nearest(self, color: Color) -> Color:
        """Find the nearest palette color to a given color."""
        return min(self.colors, key=lambda c: c.perceptual_distance(color))

    def nearest_idx(self, color: Color) -> int:
        """Find the index of the nearest palette color."""
        return min(range(len(self.colors)),
                   key=lambda i: self.colors[i].perceptual_distance(color))

    def to_numpy(self) -> np.ndarray:
        """Convert palette to [N, 3] uint8 array."""
        return np.array([c.to_rgb_tuple() for c in self.colors], dtype=np.uint8)

    def apply_to_indices(self, indices: np.ndarray) -> np.ndarray:
        """Apply palette to an index array, returns [H, W, 3] uint8."""
        palette_arr = self.to_numpy()
        return palette_arr[np.clip(indices, 0, len(self.colors) - 1)]


# ---------------------------------------------------------------------------
# Pre-defined classic palettes
# ---------------------------------------------------------------------------

def _c(h: str) -> Color:
    return Color.from_hex(h)


# PICO-8 - 16 colors, the iconic fantasy console palette
PICO8 = ColorPalette(
    name="PICO-8",
    description="16-color palette from the PICO-8 fantasy console",
    colors=[
        _c("000000"), _c("1d2b53"), _c("7e2553"), _c("008751"),
        _c("ab5236"), _c("5f574f"), _c("c2c3c7"), _c("fff1e8"),
        _c("ff004d"), _c("ffa300"), _c("ffec27"), _c("00e436"),
        _c("29adff"), _c("83769c"), _c("ff77a8"), _c("ffccaa"),
    ],
)

# Endesga 32 - Beautiful 32-color palette by Endesga
ENDESGA32 = ColorPalette(
    name="Endesga-32",
    description="32-color palette by Endesga, widely praised for pixel art",
    colors=[
        _c("be4a2f"), _c("d77643"), _c("ead4aa"), _c("e4a672"),
        _c("b86f50"), _c("733e39"), _c("3e2731"), _c("a22633"),
        _c("e43b44"), _c("f77622"), _c("feae34"), _c("fee761"),
        _c("63c74d"), _c("3e8948"), _c("265c42"), _c("193c3e"),
        _c("124e89"), _c("0099db"), _c("2ce8f5"), _c("ffffff"),
        _c("c0cbdc"), _c("8b9bb4"), _c("5a6988"), _c("3a3f5c"),
        _c("21273f"), _c("000000"), _c("ff0044"), _c("ffffff"),
        _c("8aebf1"), _c("28ccdf"), _c("0082bc"), _c("004f73"),
    ],
)

# Sweetie 16 - Modern retro palette
SWEETIE16 = ColorPalette(
    name="Sweetie-16",
    description="Sweetie 16 by GrafxKid, balanced retro palette",
    colors=[
        _c("1a1c2c"), _c("5d275d"), _c("b13e53"), _c("ef7d57"),
        _c("ffcd75"), _c("a7f070"), _c("38b764"), _c("257179"),
        _c("29366f"), _c("3b5dc9"), _c("41a6f6"), _c("73eff7"),
        _c("f4f4f4"), _c("94b0c2"), _c("566c86"), _c("333c57"),
    ],
)

# Arne 16 - Classic pixel art palette
ARNE16 = ColorPalette(
    name="Arne-16",
    description="16-color palette by ArneColor, great for game graphics",
    colors=[
        _c("000000"), _c("9d9d9d"), _c("ffffff"), _c("be2633"),
        _c("e06f8b"), _c("493c2b"), _c("a46422"), _c("eb8931"),
        _c("f7e26b"), _c("2f484e"), _c("44891a"), _c("a3ce27"),
        _c("1b2632"), _c("005784"), _c("31a2f2"), _c("b2dcef"),
    ],
)

# NES-subset - classic NES aesthetic (selected 16 from NES 54-color palette)
NES_SUBSET = ColorPalette(
    name="NES-Classic",
    description="16 carefully selected colors from the NES palette",
    colors=[
        _c("000000"), _c("fcfcfc"), _c("f8f8f8"), _c("bcbcbc"),
        _c("7c7c7c"), _c("fc0000"), _c("d82800"), _c("940084"),
        _c("0000fc"), _c("0000bc"), _c("0028f8"), _c("00b800"),
        _c("007800"), _c("f87858"), _c("fca044"), _c("f8d878"),
    ],
)

# Game Boy (4 shades of green)
GAMEBOY = ColorPalette(
    name="Game Boy",
    description="Original Game Boy 4-shade green palette",
    colors=[
        _c("0f380f"), _c("306230"), _c("8bac0f"), _c("9bbc0f"),
    ],
)

# Zughy 32 - vibrant modern retro
ZUGHY32 = ColorPalette(
    name="Zughy-32",
    description="32-color palette by Zughy, bright and vibrant",
    colors=[
        _c("472d3c"), _c("5c3a4a"), _c("7b5059"), _c("a0737a"),
        _c("c9a0a0"), _c("f5cece"), _c("f5c4b0"), _c("d19a7d"),
        _c("a06050"), _c("6e3d35"), _c("412022"), _c("6e1c2a"),
        _c("a52535"), _c("d44040"), _c("e87050"), _c("f0a060"),
        _c("f0c878"), _c("d0a060"), _c("906830"), _c("503820"),
        _c("202838"), _c("2a3850"), _c("304870"), _c("3a6090"),
        _c("4080b8"), _c("60a8d0"), _c("90c8e0"), _c("c0e8f0"),
        _c("f0f8ff"), _c("b0c8d0"), _c("708090"), _c("485060"),
    ],
)

# Resurrect 64 - large professional palette
RESURRECT64 = ColorPalette(
    name="Resurrect-64",
    description="64-color palette by Kerrie Lake, professional quality",
    colors=[
        _c("2e222f"), _c("3e3546"), _c("625565"), _c("966c6c"),
        _c("ab947a"), _c("694f62"), _c("7f708a"), _c("9babb2"),
        _c("c7dcd0"), _c("ffffff"), _c("6e2727"), _c("b33831"),
        _c("ea4f36"), _c("f57d4a"), _c("ae2334"), _c("e83b3b"),
        _c("fb6b1d"), _c("f79617"), _c("f9c22b"), _c("7a3045"),
        _c("9e4539"), _c("cd683d"), _c("e6904e"), _c("fbb954"),
        _c("fbdf8c"), _c("e8c170"), _c("c39a56"), _c("8c6239"),
        _c("5c3a1e"), _c("3b2510"), _c("1e1a14"), _c("2b2b36"),
        _c("395476"), _c("427590"), _c("7dbcd2"), _c("aad9e8"),
        _c("dff6f5"), _c("4a9b8e"), _c("2f6e6e"), _c("1e4d4d"),
        _c("1e3748"), _c("1c4e6a"), _c("3d6e8a"), _c("5f94ac"),
        _c("9dc8d4"), _c("6ab1c8"), _c("4190b0"), _c("26638a"),
        _c("184372"), _c("102250"), _c("0f1030"), _c("291f4d"),
        _c("4d3a72"), _c("745c8c"), _c("9980b0"), _c("cab8d8"),
        _c("f0e8f8"), _c("d4c0e0"), _c("a890c0"), _c("8060a0"),
        _c("5a3d80"), _c("3c2560"), _c("221848"), _c("0d0d22"),
    ],
)

# All named palettes registry
PALETTES: dict[str, ColorPalette] = {
    "pico8": PICO8,
    "endesga32": ENDESGA32,
    "sweetie16": SWEETIE16,
    "arne16": ARNE16,
    "nes": NES_SUBSET,
    "gameboy": GAMEBOY,
    "zughy32": ZUGHY32,
    "resurrect64": RESURRECT64,
}


# ---------------------------------------------------------------------------
# Procedural palette generator
# ---------------------------------------------------------------------------

class PaletteGenerator:
    """
    Generate harmonious color palettes procedurally.

    Uses color theory principles:
    - Monochromatic: shades/tints of one hue
    - Analogous: neighboring hues (±30°)
    - Complementary: opposite hues (180°)
    - Triadic: three equidistant hues (120° apart)
    - Split-complementary: one hue + two near its complement
    - Tetradic: four hues (90° apart)
    """

    HARMONY_OFFSETS = {
        "monochromatic": [0],
        "analogous": [-30, 0, 30],
        "complementary": [0, 180],
        "triadic": [0, 120, 240],
        "split_complementary": [0, 150, 210],
        "tetradic": [0, 90, 180, 270],
    }

    @classmethod
    def generate(
        cls,
        base_hue: float = 0.0,           # 0-360 base hue
        saturation: float = 0.75,         # 0-1 saturation
        lightness_range: tuple[float, float] = (0.15, 0.85),  # min/max lightness
        num_shades: int = 4,              # shades per hue
        harmony: str = "triadic",         # harmony type
        dark_outline: bool = True,        # add very dark color for outlines
        bright_highlight: bool = True,    # add very bright highlight
        name: str = "procedural",
    ) -> ColorPalette:
        """Generate a harmonious palette from a base hue."""
        offsets = cls.HARMONY_OFFSETS.get(harmony, [0])
        l_min, l_max = lightness_range
        colors = []

        # Very dark outline color
        if dark_outline:
            colors.append(Color.from_hsl(base_hue, saturation * 0.3, 0.08))

        for hue_offset in offsets:
            hue = (base_hue + hue_offset) % 360
            for i in range(num_shades):
                t = i / max(num_shades - 1, 1)
                l = l_min + t * (l_max - l_min)
                # Vary saturation slightly with lightness (more sat at mid-tones)
                sat_mod = saturation * (1.0 - abs(l - 0.5) * 0.4)
                colors.append(Color.from_hsl(hue, sat_mod, l))

        # Bright highlight (near white but tinted)
        if bright_highlight:
            colors.append(Color.from_hsl(base_hue, 0.3, 0.93))

        return ColorPalette(name=name, colors=colors,
                            description=f"Procedural {harmony} palette, base hue {base_hue:.0f}°")

    @classmethod
    def generate_sprite_palette(
        cls,
        body_hue: float,
        accent_hue: float,
        seed: Optional[int] = None,
    ) -> ColorPalette:
        """
        Generate a palette specifically for sprite coloring.

        Returns 8 colors in a consistent semantic layout:
        Index 0: Transparent placeholder (black, used as alpha key)
        Index 1: Deep outline/shadow
        Index 2: Dark body
        Index 3: Main body mid-tone
        Index 4: Light body highlight
        Index 5: Bright highlight/sheen
        Index 6: Accent dark (eye, gem, etc.)
        Index 7: Accent bright (eye shine, glow)
        """
        rng = random.Random(seed)
        # Add slight randomness to hues
        body_hue = (body_hue + rng.uniform(-15, 15)) % 360
        accent_hue = (accent_hue + rng.uniform(-10, 10)) % 360

        body_sat = rng.uniform(0.60, 0.90)
        accent_sat = rng.uniform(0.80, 1.0)

        colors = [
            Color(0, 0, 0, 0),                              # 0: transparent
            Color.from_hsl(body_hue, body_sat * 0.8, 0.10),  # 1: deep outline
            Color.from_hsl(body_hue, body_sat, 0.22),         # 2: dark body
            Color.from_hsl(body_hue, body_sat, 0.42),         # 3: main body
            Color.from_hsl(body_hue, body_sat * 0.9, 0.62),   # 4: light body
            Color.from_hsl(body_hue, 0.30, 0.88),             # 5: bright highlight
            Color.from_hsl(accent_hue, accent_sat, 0.30),     # 6: accent dark
            Color.from_hsl(accent_hue, accent_sat, 0.65),     # 7: accent bright
        ]
        return ColorPalette(
            name=f"sprite_{body_hue:.0f}_{accent_hue:.0f}",
            colors=colors,
            description=f"Sprite palette body={body_hue:.0f}° accent={accent_hue:.0f}°",
        )

    @classmethod
    def from_preset(cls, name: str) -> ColorPalette:
        """Load a named preset palette."""
        name_lower = name.lower().replace("-", "").replace("_", "").replace(" ", "")
        # Fuzzy match
        for key, palette in PALETTES.items():
            if key.replace("-", "") == name_lower:
                return palette
        raise KeyError(f"Unknown palette '{name}'. Available: {list(PALETTES.keys())}")

    @classmethod
    def randomize(
        cls,
        seed: int,
        num_colors: int = 16,
        harmony: str = "triadic",
    ) -> ColorPalette:
        """Generate a random but harmonious palette from a seed."""
        rng = random.Random(seed)
        base_hue = rng.uniform(0, 360)
        sat = rng.uniform(0.55, 0.90)
        num_shades = max(2, num_colors // len(cls.HARMONY_OFFSETS.get(harmony, [0])))
        return cls.generate(
            base_hue=base_hue,
            saturation=sat,
            num_shades=num_shades,
            harmony=harmony,
            name=f"random_seed{seed}",
        )


# ---------------------------------------------------------------------------
# Sprite-specific palettes for our game themes
# ---------------------------------------------------------------------------

# Galaga color sets (alien hue, accent hue) → sprite palette
GALAGA_PALETTES = {
    "bee_purple":    (280, 55),   # purple bee with yellow eyes
    "bee_blue":      (220, 50),   # blue variant
    "bee_green":     (140, 30),   # green variant
    "dragonfly_red": (5, 190),    # red dragonfly with cyan eyes
    "boss_gold":     (45, 270),   # gold boss alien
    "boss_cyan":     (185, 30),   # cyan boss variant
    "player_cyan":   (195, 50),   # player ship (cool blue)
    "explosion":     (30, 60),    # explosion (orange/yellow)
}

# Zelda color sets
ZELDA_PALETTES = {
    "link_green":    (125, 45),   # classic Link green
    "link_blue":     (215, 45),   # blue Link
    "fairy":         (170, 300),  # fairy/Navi
    "rupee_green":   (130, 60),   # green rupee
    "rupee_blue":    (210, 60),   # blue rupee
    "rupee_red":     (0, 60),     # red rupee
    "heart":         (0, 280),    # heart container red
    "octorok":       (300, 60),   # octorok pink/red
    "slime":         (150, 290),  # slime/Zol green-purple
    "darknut":       (210, 30),   # darknut blue
}

# Pacman color sets
PACMAN_PALETTES = {
    "pacman":        (50, 20),    # Pac-Man yellow
    "blinky":        (0, 50),     # red ghost
    "pinky":         (325, 195),  # pink ghost
    "inky":          (185, 320),  # cyan ghost
    "clyde":         (30, 130),   # orange ghost
    "scared":        (240, 30),   # scared ghost (blue)
    "pellet":        (50, 0),     # power pellet
    "cherry":        (355, 125),  # cherry bonus
    "maze":          (240, 60),   # maze blue
}

ALL_SPRITE_PALETTES = {**GALAGA_PALETTES, **ZELDA_PALETTES, **PACMAN_PALETTES}


# ---------------------------------------------------------------------------
# Tree-specific palettes
# ---------------------------------------------------------------------------
# Trees use a different semantic index layout than character sprites:
#   Index 1-3: bark colors (dark -> light)
#   Index 4-6: leaf/needle colors (dark -> highlight)
#   Index 7:   special (snow, blossoms, birch bark marks, etc.)
#
# Config format: (leaf_hue, bark_hue, special_color_or_None)

def _snow() -> Color:
    return Color.from_hsl(200, 0.25, 0.92)

def _blossom() -> Color:
    return Color.from_hsl(340, 0.75, 0.78)

def _birch_mark() -> Color:
    return Color.from_hsl(25, 0.12, 0.18)

def _gray_highlight() -> Color:
    return Color.from_hsl(200, 0.08, 0.60)


TREE_PALETTE_CONFIGS: dict[str, tuple] = {
    # name: (leaf_hue, bark_hue, special_color_or_None)
    "pine_summer":   (130, 30,  None),        # forest green needles, warm brown bark
    "pine_winter":   (128, 30,  _snow()),     # dark green under snow
    "maple_summer":  (112, 30,  None),        # medium green canopy, brown trunk
    "maple_autumn":  (18,  28,  None),        # orange-red-brown canopy
    "maple_spring":  (135, 32,  _blossom()),  # pale green + pink blossoms
    "birch_summer":  (112, 52,  _birch_mark()),   # green canopy, cream/tan bark + marks
    "birch_autumn":  (48,  52,  _birch_mark()),   # golden canopy, cream bark + marks
    "dead_tree":     (28,  26,  _gray_highlight()),  # bare brown branches, no leaves
    # --- new tree types ---
    "willow_summer":  (128, 40,  None),                  # yellow-green drooping, warm tan bark
    "willow_spring":  (130, 40,  _blossom()),            # pale green + spring blossoms
    "spruce_summer":  (148, 18,  None),                  # blue-green conifer, dark gray bark
    "spruce_winter":  (148, 18,  _snow()),               # blue-green under snow
    "cherry_spring":  (128, 30,  _blossom()),            # green leafout + pink blossoms
    "cherry_summer":  (120, 30,  None),                  # full green canopy, brown bark
    "acacia_summer":  (70,  38,  None),                  # olive/golden-green, tan bark
    "acacia_dry":     (52,  36,  None),                  # dry season yellowish-brown
    "shrub_summer":   (118, 32,  None),                  # medium green bush, dark bark
    "shrub_autumn":   (28,  30,  None),                  # warm orange-brown autumn shrub
    "palm_tropical":  (138, 42,  None),                  # bright tropical green, tan trunk
    "palm_dry":       (62,  44,  None),                  # dry season golden fronds
    "fir_summer":     (150, 22,  None),                  # deep blue-green fir, dark bark
    "fir_winter":     (150, 22,  _snow()),               # fir under winter snow
    "apple_summer":   (115, 28,  None),                  # green apple tree, warm bark
    "apple_autumn":   (20,  28,  Color.from_hsl(0, 0.78, 0.44)),  # autumn + red fruit
    "cypress_summer": (156, 20,  None),                  # very dark narrow green, gray bark
    "cypress_winter": (156, 20,  _snow()),               # narrow cypress with snow
}


def generate_tree_palette(
    leaf_hue: float,
    bark_hue: float = 30.0,
    special_color: Optional[Color] = None,
    seed: Optional[int] = None,
) -> ColorPalette:
    """
    Generate a palette for tree sprites.

    Index semantics for trees:
      0: transparent
      1: bark dark     (darkest bark / branch shadow)
      2: bark mid      (main bark tone)
      3: bark light    (bark highlight / lighter branch)
      4: leaf dark     (deep shadow in foliage)
      5: leaf mid      (main leaf / needle tone)
      6: leaf highlight(bright leaf top / needle tip)
      7: special       (snow=white-blue, blossom=pink, birch mark=dark)
    """
    rng = random.Random(seed)
    bark_sat = rng.uniform(0.32, 0.52) + rng.uniform(-0.04, 0.04)
    leaf_sat = rng.uniform(0.55, 0.82) + rng.uniform(-0.05, 0.05)
    # Slight hue jitter for variation
    leaf_hue = (leaf_hue + rng.uniform(-8, 8)) % 360
    bark_hue = (bark_hue + rng.uniform(-5, 5)) % 360

    colors = [
        Color(0, 0, 0, 0),                                          # 0: transparent
        Color.from_hsl(bark_hue, bark_sat, 0.13),                   # 1: bark dark
        Color.from_hsl(bark_hue, bark_sat, 0.28),                   # 2: bark mid
        Color.from_hsl(bark_hue, bark_sat * 0.75, 0.48),            # 3: bark light
        Color.from_hsl(leaf_hue, leaf_sat, 0.20),                   # 4: leaf dark
        Color.from_hsl(leaf_hue, leaf_sat, 0.38),                   # 5: leaf mid
        Color.from_hsl(leaf_hue, leaf_sat * 0.88, 0.58),            # 6: leaf highlight
        special_color if special_color is not None                   # 7: special
            else Color.from_hsl(leaf_hue, leaf_sat * 0.45, 0.80),
    ]
    return ColorPalette(
        name=f"tree_{leaf_hue:.0f}_{bark_hue:.0f}",
        colors=colors,
        description=f"Tree palette leaf={leaf_hue:.0f}deg bark={bark_hue:.0f}deg",
    )


def get_sprite_palette(sprite_type: str, seed: Optional[int] = None) -> ColorPalette:
    """Get a palette for a specific sprite type with optional randomization."""
    if sprite_type in TREE_PALETTE_CONFIGS:
        leaf_hue, bark_hue, special = TREE_PALETTE_CONFIGS[sprite_type]
        return generate_tree_palette(leaf_hue, bark_hue, special, seed=seed)
    elif sprite_type in ALL_SPRITE_PALETTES:
        body_hue, accent_hue = ALL_SPRITE_PALETTES[sprite_type]
        return PaletteGenerator.generate_sprite_palette(body_hue, accent_hue, seed=seed)
    else:
        # Fall back to random harmonious palette
        hue = hash(sprite_type) % 360
        return PaletteGenerator.generate_sprite_palette(hue, (hue + 120) % 360, seed=seed)


def generate_palette_variations(
    sprite_type: str,
    n_variations: int = 8,
    base_seed: int = 0,
) -> list[ColorPalette]:
    """Generate N color variations for a sprite type."""
    return [
        get_sprite_palette(sprite_type, seed=base_seed + i)
        for i in range(n_variations)
    ]
