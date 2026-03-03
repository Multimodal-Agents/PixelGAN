"""
Pixel art sprite generator for PixelGAN training data.

Sprite categories & naming:
  gal_*   - Galaga universe (ships, aliens, effects)
  zel_*   - Zelda universe (characters, enemies, items)
  pac_*   - Pac-Man universe (pac, ghosts, fruits, maze)
  item_*  - Generic game items (coins, gems, chests, keys)
  tree_*  - Nature / trees  (see tree_generator.py for procedural variants)

Color layer index semantics (same for all sprites):
  0 = transparent
  1 = deep outline / darkest shadow
  2 = dark body / inner shadow
  3 = main body mid-tone
  4 = light body / highlight
  5 = bright highlight / sheen
  6 = accent dark  (eyes, gems, special feature)
  7 = accent bright (eye shine, glow, highlight dot)

Each sprite registered with _reg() gets a category, native size, and a
palette_type hint that drives procedural color variation.
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .color_palette import (
    ColorPalette, PaletteGenerator,
    get_sprite_palette, generate_palette_variations,
    SWEETIE16,
)
from .dithering import apply_dithering


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SPRITES: dict[str, dict] = {}

def _reg(name: str, category: str, size: int, data: list[list[int]],
         palette_type: str, description: str = "") -> None:
    assert len(data) == size, f"{name}: {len(data)} rows != {size}"
    for i, row in enumerate(data):
        assert len(row) == size, f"{name} row {i}: {len(row)} cols != {size}"
    SPRITES[name] = dict(category=category, size=size, data=data,
                         palette_type=palette_type, description=description)


# ============================================================
#  GALAGA  (gal_*)
# ============================================================

# Player fighter ship
_reg("gal_fighter", "galaga", 8, [
    [0, 0, 0, 1, 1, 0, 0, 0],  # nose tip
    [0, 0, 1, 5, 5, 1, 0, 0],  # cockpit glass (shine)
    [0, 1, 3, 4, 4, 3, 1, 0],  # fuselage
    [1, 3, 4, 5, 5, 4, 3, 1],  # wide hull
    [1, 3, 3, 3, 3, 3, 3, 1],  # lower hull
    [1, 2, 3, 2, 2, 3, 2, 1],  # engines
    [0, 1, 2, 0, 0, 2, 1, 0],  # exhaust
    [0, 0, 0, 0, 0, 0, 0, 0],
], "player_cyan", "Player fighter - Galaga bee-fighter style")

# Bee alien type A (antennae + round eyes)
_reg("gal_bee_a", "galaga", 8, [
    [0, 1, 0, 0, 0, 0, 1, 0],  # antennae
    [0, 1, 3, 6, 6, 3, 1, 0],  # head + eye sockets
    [1, 3, 7, 6, 6, 7, 3, 1],  # eyes with iris shine
    [1, 4, 3, 3, 3, 3, 4, 1],  # shoulder highlights
    [1, 3, 3, 3, 3, 3, 3, 1],  # main body
    [1, 3, 3, 1, 1, 3, 3, 1],  # underbelly markings
    [0, 1, 3, 3, 3, 3, 1, 0],  # abdomen
    [1, 0, 1, 0, 0, 1, 0, 1],  # claws
], "bee_purple", "Bee alien A - regular Galaga formation enemy")

# Bee alien type B (double antenna, different posture)
_reg("gal_bee_b", "galaga", 8, [
    [0, 1, 0, 1, 1, 0, 1, 0],  # double antenna
    [1, 3, 1, 6, 6, 1, 3, 1],  # eyes with ring
    [1, 3, 7, 6, 6, 7, 3, 1],  # eyes
    [1, 1, 3, 3, 3, 3, 1, 1],  # wide shoulder
    [0, 3, 3, 3, 3, 3, 3, 0],  # body
    [0, 1, 3, 4, 4, 3, 1, 0],  # highlight belly
    [0, 1, 1, 3, 3, 1, 1, 0],  # lower
    [1, 0, 0, 1, 1, 0, 0, 1],  # feet
], "bee_blue", "Bee alien B - second formation row")

# Bee alien type C (compact, aggressive looking)
_reg("gal_bee_c", "galaga", 8, [
    [0, 0, 1, 0, 0, 1, 0, 0],  # short antennae
    [0, 1, 3, 1, 1, 3, 1, 0],
    [1, 3, 6, 7, 7, 6, 3, 1],  # big eyes
    [1, 3, 3, 3, 3, 3, 3, 1],
    [1, 4, 3, 3, 3, 3, 4, 1],  # highlights
    [0, 1, 3, 3, 3, 3, 1, 0],
    [0, 1, 2, 3, 3, 2, 1, 0],
    [0, 1, 0, 1, 1, 0, 1, 0],  # feet
], "bee_green", "Bee alien C - tier-3 formation enemy")

# Dragonfly alien (mid-tier, 8x8)
_reg("gal_dragonfly", "galaga", 8, [
    [1, 0, 3, 3, 3, 3, 0, 1],  # wing tips
    [1, 3, 4, 5, 5, 4, 3, 1],  # wings with shine
    [0, 1, 3, 6, 6, 3, 1, 0],  # head
    [0, 1, 7, 6, 6, 7, 1, 0],  # eyes
    [0, 1, 3, 3, 3, 3, 1, 0],  # body upper
    [0, 1, 3, 4, 4, 3, 1, 0],  # body mid
    [0, 1, 1, 3, 3, 1, 1, 0],  # body lower
    [0, 0, 1, 0, 0, 1, 0, 0],  # stinger
], "dragonfly_red", "Dragonfly alien - Galaga midfield guard")

# Boss Galaga flagship (butterfly shape, 16x16) - IMPROVED
_reg("gal_flagship", "galaga", 16, [
    [0, 0, 1, 3, 3, 0, 0, 0, 0, 0, 0, 3, 3, 1, 0, 0],  # wing tips
    [0, 1, 3, 4, 4, 3, 1, 0, 0, 1, 3, 4, 4, 3, 1, 0],  # upper wings
    [1, 3, 4, 5, 5, 4, 3, 1, 1, 3, 4, 5, 5, 4, 3, 1],  # wing highlight
    [1, 3, 3, 3, 3, 3, 3, 1, 1, 3, 3, 3, 3, 3, 3, 1],  # wing body
    [1, 3, 3, 6, 6, 3, 3, 3, 3, 3, 3, 6, 6, 3, 3, 1],  # wing + eye zone
    [1, 4, 3, 7, 6, 3, 1, 3, 3, 1, 3, 6, 7, 3, 4, 1],  # eye pupils + wing
    [1, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 1],  # wide mid
    [0, 1, 3, 3, 4, 3, 3, 3, 3, 3, 3, 4, 3, 3, 1, 0],  # highlight band
    [0, 0, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 0, 0],
    [0, 0, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 0, 0],
    [0, 1, 3, 3, 3, 2, 3, 3, 3, 3, 2, 3, 3, 3, 1, 0],  # lower markings
    [0, 1, 3, 3, 2, 2, 2, 3, 3, 2, 2, 2, 3, 3, 1, 0],
    [0, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 0],
    [1, 3, 3, 3, 3, 1, 0, 0, 0, 0, 1, 3, 3, 3, 3, 1],  # lower wing split
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], "boss_gold", "Flagship boss - butterfly form (16x16)")

# Mystery ship (horizontal flier)
_reg("gal_mystery", "galaga", 8, [
    [0, 0, 1, 1, 1, 1, 0, 0],
    [0, 1, 3, 5, 5, 3, 1, 0],
    [1, 3, 5, 7, 7, 5, 3, 1],  # shiny dome
    [1, 4, 3, 5, 5, 3, 4, 1],
    [1, 3, 6, 3, 3, 6, 3, 1],  # port lights
    [1, 3, 3, 3, 3, 3, 3, 1],
    [0, 1, 2, 3, 3, 2, 1, 0],
    [0, 0, 1, 1, 1, 1, 0, 0],
], "boss_cyan", "Mystery ship - bonus target")

# Explosion frame
_reg("gal_explosion", "galaga", 8, [
    [0, 0, 0, 6, 6, 0, 0, 0],
    [0, 0, 7, 7, 7, 7, 0, 0],
    [0, 7, 5, 5, 5, 5, 7, 0],
    [6, 7, 5, 4, 4, 5, 7, 6],
    [6, 7, 5, 4, 4, 5, 7, 6],
    [0, 7, 5, 5, 5, 5, 7, 0],
    [0, 0, 7, 7, 7, 7, 0, 0],
    [0, 0, 0, 6, 6, 0, 0, 0],
], "explosion", "Explosion burst")

# Laser bullet
_reg("gal_bullet", "galaga", 8, [
    [0, 0, 0, 5, 5, 0, 0, 0],
    [0, 0, 0, 5, 5, 0, 0, 0],
    [0, 0, 0, 7, 7, 0, 0, 0],
    [0, 0, 1, 4, 4, 1, 0, 0],
    [0, 0, 1, 4, 4, 1, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
], "player_cyan", "Player laser bolt")

# Tractor beam (for flagship capture)
_reg("gal_beam", "galaga", 8, [
    [0, 0, 0, 7, 7, 0, 0, 0],
    [0, 0, 7, 5, 5, 7, 0, 0],
    [0, 0, 5, 4, 4, 5, 0, 0],
    [0, 5, 4, 3, 3, 4, 5, 0],
    [0, 5, 4, 3, 3, 4, 5, 0],
    [0, 0, 5, 4, 4, 5, 0, 0],
    [0, 0, 7, 5, 5, 7, 0, 0],
    [0, 0, 0, 7, 7, 0, 0, 0],
], "boss_cyan", "Tractor beam pulse")


# ============================================================
#  ZELDA  (zel_*)
# ============================================================

# --- Link redesigned: chibi proportions, clear hat, cute eyes ---
# Rows 0-5: pointed green hat (dominant feature)
# Rows 6-9: face (skin, big eyes, mouth)
# Rows 10-12: tunic body with belt
# Rows 13-15: boots/legs
_reg("zel_link", "zelda", 16, [
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # hat tip (centered)
    [0, 0, 0, 0, 0, 0, 1, 3, 3, 1, 0, 0, 0, 0, 0, 0],  # hat narrow
    [0, 0, 0, 0, 0, 1, 3, 4, 3, 3, 1, 0, 0, 0, 0, 0],  # hat + fold highlight
    [0, 0, 0, 0, 1, 3, 3, 4, 3, 3, 3, 1, 0, 0, 0, 0],  # hat wider
    [0, 0, 0, 1, 3, 3, 3, 4, 3, 3, 3, 3, 1, 0, 0, 0],  # hat wide
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],  # hat brim (solid bar)
    [0, 0, 0, 1, 5, 5, 5, 5, 5, 5, 5, 5, 1, 0, 0, 0],  # forehead (skin)
    [0, 0, 0, 1, 5, 5, 6, 5, 5, 6, 5, 5, 1, 0, 0, 0],  # EYES (cols 6 & 9)
    [0, 0, 0, 1, 5, 5, 5, 5, 5, 5, 5, 5, 1, 0, 0, 0],  # nose area
    [0, 0, 0, 0, 1, 5, 5, 7, 7, 5, 5, 1, 0, 0, 0, 0],  # mouth (small)
    [0, 0, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 0, 0],  # tunic upper
    [0, 0, 1, 3, 3, 4, 3, 2, 2, 3, 4, 3, 3, 1, 0, 0],  # belt buckle
    [0, 0, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1, 0, 0],  # tunic lower shadow
    [0, 0, 0, 1, 3, 3, 1, 0, 0, 1, 3, 3, 1, 0, 0, 0],  # legs split
    [0, 0, 0, 1, 2, 3, 1, 0, 0, 1, 3, 2, 1, 0, 0, 0],  # boots
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],  # boot toes
], "link_green", "Link - chibi top-down, pointed hat, cute eyes (16x16)")

# Link with sword drawn (attack pose, 16x16)
_reg("zel_link_sword", "zelda", 16, [
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 5, 0, 0],  # hat + sword tip
    [0, 0, 0, 0, 0, 0, 1, 3, 3, 1, 0, 0, 0, 5, 0, 0],
    [0, 0, 0, 0, 0, 1, 3, 4, 3, 3, 1, 0, 0, 4, 0, 0],
    [0, 0, 0, 0, 1, 3, 3, 4, 3, 3, 3, 1, 0, 4, 0, 0],
    [0, 0, 0, 1, 3, 3, 3, 4, 3, 3, 3, 3, 1, 3, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],  # hat brim
    [0, 0, 0, 1, 5, 5, 5, 5, 5, 5, 5, 5, 1, 0, 0, 0],
    [0, 0, 0, 1, 5, 5, 6, 5, 5, 6, 5, 5, 1, 0, 0, 0],  # eyes
    [0, 0, 0, 1, 5, 5, 5, 5, 5, 5, 5, 5, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 5, 5, 7, 7, 5, 5, 1, 0, 0, 0, 0],
    [0, 0, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 0, 0],
    [0, 0, 1, 3, 3, 4, 3, 2, 2, 3, 4, 3, 3, 1, 0, 0],
    [0, 0, 1, 2, 3, 3, 3, 5, 3, 3, 3, 3, 2, 1, 0, 0],  # arm extended right
    [0, 0, 0, 1, 3, 3, 1, 0, 0, 1, 3, 3, 6, 6, 1, 0],  # sword hilt right
    [0, 0, 0, 1, 2, 3, 1, 0, 0, 1, 3, 2, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
], "link_blue", "Link - sword drawn attack pose (16x16)")

# Octorok (rock-spitting octopus enemy)
_reg("zel_octorok", "zelda", 8, [
    [0, 1, 1, 1, 1, 1, 1, 0],  # shell top
    [1, 3, 4, 3, 3, 4, 3, 1],  # shell sheen
    [1, 3, 3, 6, 6, 3, 3, 1],  # eyes
    [1, 3, 7, 6, 6, 7, 3, 1],  # eye shine
    [1, 3, 3, 3, 3, 3, 3, 1],  # body
    [0, 1, 3, 3, 3, 3, 1, 0],  # lower
    [0, 1, 3, 2, 2, 3, 1, 0],  # tentacles/legs
    [0, 0, 1, 1, 1, 1, 0, 0],
], "octorok", "Octorok - rock-spitting blob enemy")

# Keese (bat enemy, top-down)
_reg("zel_keese", "zelda", 8, [
    [1, 3, 3, 0, 0, 3, 3, 1],  # wing spread
    [3, 4, 3, 1, 1, 3, 4, 3],  # wing highlight
    [3, 3, 1, 6, 6, 1, 3, 3],  # eyes
    [0, 1, 3, 6, 6, 3, 1, 0],  # face
    [0, 1, 3, 3, 3, 3, 1, 0],  # body
    [0, 1, 3, 2, 2, 3, 1, 0],
    [0, 0, 1, 3, 3, 1, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0],  # tail
], "bee_purple", "Keese - bat enemy")

# Slime / Zol enemy
_reg("zel_slime", "zelda", 8, [
    [0, 1, 1, 1, 1, 1, 0, 0],
    [1, 3, 4, 4, 3, 3, 1, 0],
    [1, 3, 4, 5, 4, 3, 3, 1],
    [1, 3, 6, 3, 6, 3, 3, 1],  # eyes
    [1, 3, 7, 3, 7, 3, 3, 1],  # eye shine
    [1, 3, 3, 3, 3, 3, 3, 1],
    [0, 1, 3, 2, 2, 3, 1, 0],
    [0, 0, 1, 1, 1, 1, 0, 0],
], "slime", "Slime - Zol blob enemy")

# Moblin (pig warrior enemy, 16x16)
_reg("zel_moblin", "zelda", 16, [
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],  # head top
    [0, 0, 0, 1, 3, 3, 3, 3, 3, 3, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 3, 3, 3, 3, 3, 3, 3, 3, 1, 0, 0, 0, 0],
    [0, 0, 1, 3, 6, 3, 3, 3, 6, 3, 3, 1, 0, 0, 0, 0],  # eyes
    [0, 0, 1, 3, 3, 3, 3, 3, 3, 3, 3, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 5, 5, 3, 3, 5, 5, 1, 0, 0, 0, 0, 0],  # snout
    [0, 0, 0, 1, 5, 6, 3, 3, 6, 5, 1, 0, 0, 0, 0, 0],  # nostrils
    [0, 0, 0, 0, 1, 3, 7, 7, 3, 1, 0, 0, 0, 0, 0, 0],  # fangs
    [0, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 0, 0, 0, 0],  # shoulders
    [0, 1, 3, 3, 4, 3, 3, 3, 3, 4, 3, 1, 0, 0, 0, 0],  # armor glint
    [0, 1, 2, 3, 3, 3, 3, 3, 3, 3, 2, 1, 0, 0, 0, 0],
    [0, 0, 1, 2, 3, 3, 3, 3, 3, 2, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 3, 3, 1, 3, 3, 1, 3, 1, 0, 0, 0, 0, 0],  # legs
    [0, 0, 1, 2, 3, 1, 0, 1, 3, 2, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 2, 3, 1, 0, 1, 3, 2, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
], "octorok", "Moblin - pig warrior enemy (16x16)")

# Rupee (green by default, palette drives color)
_reg("zel_rupee", "zelda", 8, [
    [0, 0, 1, 1, 1, 1, 0, 0],
    [0, 1, 3, 5, 4, 3, 1, 0],
    [1, 3, 5, 5, 4, 4, 3, 1],
    [1, 3, 5, 4, 4, 3, 3, 1],
    [1, 3, 4, 3, 3, 3, 3, 1],
    [1, 3, 3, 3, 3, 3, 3, 1],
    [0, 1, 3, 3, 3, 3, 1, 0],
    [0, 0, 1, 1, 1, 1, 0, 0],
], "rupee_green", "Green Rupee - most common currency")

_reg("zel_rupee_blue", "zelda", 8, [
    [0, 0, 1, 1, 1, 1, 0, 0],
    [0, 1, 3, 5, 4, 3, 1, 0],
    [1, 3, 5, 7, 4, 4, 3, 1],
    [1, 3, 5, 4, 4, 3, 3, 1],
    [1, 3, 4, 3, 3, 3, 3, 1],
    [1, 3, 3, 3, 3, 3, 3, 1],
    [0, 1, 3, 3, 3, 3, 1, 0],
    [0, 0, 1, 1, 1, 1, 0, 0],
], "rupee_blue", "Blue Rupee - medium value")

_reg("zel_rupee_red", "zelda", 8, [
    [0, 0, 1, 1, 1, 1, 0, 0],
    [0, 1, 3, 5, 4, 3, 1, 0],
    [1, 3, 5, 7, 4, 4, 3, 1],
    [1, 3, 5, 4, 4, 3, 3, 1],
    [1, 3, 4, 3, 3, 3, 3, 1],
    [1, 3, 3, 3, 3, 3, 3, 1],
    [0, 1, 3, 3, 3, 3, 1, 0],
    [0, 0, 1, 1, 1, 1, 0, 0],
], "rupee_red", "Red Rupee - high value")

# Heart container
_reg("zel_heart", "zelda", 8, [
    [0, 1, 1, 0, 0, 1, 1, 0],
    [1, 3, 3, 1, 1, 3, 3, 1],
    [1, 4, 5, 3, 3, 5, 3, 1],
    [1, 3, 3, 3, 3, 3, 3, 1],
    [0, 1, 3, 3, 3, 3, 1, 0],
    [0, 0, 1, 3, 3, 1, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
], "heart", "Heart container - life pickup")

# Fairy / Navi companion
_reg("zel_fairy", "zelda", 8, [
    [0, 1, 3, 0, 0, 3, 1, 0],  # wings top
    [1, 3, 4, 1, 1, 4, 3, 1],  # wings spread
    [0, 1, 3, 3, 3, 3, 1, 0],  # body glow aura
    [0, 0, 1, 5, 5, 1, 0, 0],  # face
    [0, 0, 1, 7, 7, 1, 0, 0],  # sparkle
    [0, 1, 3, 4, 4, 3, 1, 0],  # body
    [0, 1, 3, 3, 3, 3, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0],
], "fairy", "Fairy - companion / life-restore item")

# Sword (diagonally oriented)
_reg("zel_sword", "zelda", 8, [
    [0, 0, 0, 0, 0, 5, 1, 0],
    [0, 0, 0, 0, 5, 4, 1, 0],
    [0, 0, 0, 5, 4, 3, 1, 0],
    [0, 0, 5, 4, 3, 3, 1, 0],
    [0, 5, 6, 3, 1, 1, 0, 0],  # guard
    [5, 6, 6, 7, 1, 0, 0, 0],  # guard ornament
    [0, 3, 3, 3, 1, 0, 0, 0],  # handle
    [0, 0, 2, 2, 1, 0, 0, 0],  # pommel
], "rupee_blue", "Sword - blade weapon item")

# Shield (kite shield, 8x8)
_reg("zel_shield", "zelda", 8, [
    [0, 1, 1, 1, 1, 1, 0, 0],
    [1, 3, 5, 5, 3, 3, 1, 0],
    [1, 3, 5, 6, 3, 3, 3, 1],
    [1, 3, 3, 6, 3, 3, 3, 1],  # emblem
    [1, 3, 3, 3, 3, 3, 3, 1],
    [0, 1, 3, 3, 3, 3, 1, 0],
    [0, 0, 1, 3, 3, 1, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0],
], "bee_blue", "Shield - defensive item")

# Boomerang
_reg("zel_boomerang", "zelda", 8, [
    [0, 0, 1, 3, 3, 3, 1, 0],
    [0, 1, 3, 4, 3, 3, 3, 1],
    [1, 3, 4, 3, 3, 3, 1, 0],
    [1, 3, 3, 3, 1, 1, 0, 0],
    [0, 1, 3, 1, 3, 0, 0, 0],
    [0, 0, 1, 3, 3, 1, 0, 0],
    [0, 0, 0, 1, 3, 3, 1, 0],
    [0, 0, 0, 0, 1, 1, 0, 0],
], "rupee_red", "Boomerang - thrown weapon")

# Bomb
_reg("zel_bomb", "zelda", 8, [
    [0, 0, 0, 0, 1, 0, 0, 0],  # fuse
    [0, 0, 0, 0, 7, 0, 0, 0],  # fuse spark
    [0, 0, 1, 1, 1, 1, 0, 0],
    [0, 1, 3, 3, 3, 3, 1, 0],
    [1, 3, 3, 2, 3, 3, 3, 1],
    [1, 3, 2, 2, 3, 3, 3, 1],
    [0, 1, 3, 3, 3, 3, 1, 0],
    [0, 0, 1, 1, 1, 1, 0, 0],
], "bee_purple", "Bomb - explosive item")

# Treasure chest (8x8)
_reg("zel_chest", "zelda", 8, [
    [0, 1, 1, 1, 1, 1, 1, 0],  # lid top
    [1, 3, 3, 4, 4, 3, 3, 1],  # lid face
    [1, 3, 6, 7, 7, 6, 3, 1],  # lock/clasp
    [1, 1, 1, 1, 1, 1, 1, 1],  # lid-body seam
    [1, 2, 3, 3, 3, 3, 2, 1],  # body
    [1, 2, 3, 3, 3, 3, 2, 1],
    [1, 2, 3, 3, 3, 3, 2, 1],
    [0, 1, 2, 2, 2, 2, 1, 0],  # base
], "boss_gold", "Treasure chest - collectible container")

# Key
_reg("zel_key", "zelda", 8, [
    [0, 0, 1, 1, 1, 0, 0, 0],  # bow (ring) top
    [0, 1, 3, 3, 3, 1, 0, 0],
    [0, 1, 3, 6, 3, 1, 0, 0],  # hole in key bow
    [0, 1, 3, 3, 3, 1, 0, 0],
    [0, 0, 1, 3, 1, 0, 0, 0],  # shaft starts
    [0, 0, 1, 3, 1, 1, 0, 0],  # first bit
    [0, 0, 1, 3, 1, 1, 1, 0],  # second bit
    [0, 0, 0, 1, 1, 0, 0, 0],  # tip
], "boss_gold", "Key - door opener item")

# Triforce (16x16 - iconic!)
_reg("zel_triforce", "zelda", 16, [
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 3, 3, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 3, 7, 4, 3, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 3, 3, 4, 7, 3, 3, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 3, 3, 3, 3, 3, 3, 3, 3, 1, 0, 0, 0],
    [0, 0, 1, 3, 3, 1, 0, 0, 0, 0, 1, 3, 3, 1, 0, 0],  # gap between triangles
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
    [1, 3, 3, 1, 0, 0, 1, 3, 3, 1, 0, 0, 1, 3, 3, 1],  # two bottom triangles
    [1, 3, 7, 3, 1, 1, 3, 7, 4, 3, 1, 1, 3, 4, 7, 1],
    [1, 3, 4, 3, 3, 3, 3, 4, 7, 3, 3, 3, 3, 7, 4, 1],
    [1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1],
    [0, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 0],
    [0, 0, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 0, 0],
    [0, 0, 0, 1, 3, 3, 3, 3, 3, 3, 3, 3, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
], "boss_gold", "Triforce - sacred golden artifact (16x16)")


# ============================================================
#  PAC-MAN  (pac_*)
# ============================================================

_reg("pac_pacman_open", "pacman", 8, [
    [0, 1, 1, 1, 1, 1, 0, 0],
    [1, 3, 4, 3, 3, 3, 1, 0],
    [1, 4, 5, 3, 3, 0, 0, 0],  # mouth open
    [1, 3, 3, 3, 0, 0, 0, 0],
    [1, 3, 3, 3, 0, 0, 0, 0],
    [1, 4, 3, 3, 3, 0, 0, 0],
    [1, 3, 3, 3, 3, 3, 1, 0],
    [0, 1, 1, 1, 1, 1, 0, 0],
], "pacman", "Pac-Man - open mouth chomping")

_reg("pac_pacman_closed", "pacman", 8, [
    [0, 1, 1, 1, 1, 1, 1, 0],
    [1, 3, 4, 3, 3, 3, 3, 1],
    [1, 4, 5, 3, 3, 3, 3, 1],
    [1, 3, 3, 3, 3, 3, 3, 1],
    [1, 3, 3, 3, 3, 3, 3, 1],
    [1, 4, 3, 3, 3, 3, 3, 1],
    [1, 3, 3, 3, 3, 3, 3, 1],
    [0, 1, 1, 1, 1, 1, 1, 0],
], "pacman", "Pac-Man - closed mouth")

# All 4 ghosts share same body template, palette drives color
def _ghost(palette_type: str, name: str, desc: str):
    _reg(f"pac_{name}", "pacman", 8, [
        [0, 1, 1, 1, 1, 1, 1, 0],
        [1, 3, 3, 4, 3, 4, 3, 1],
        [1, 3, 7, 6, 7, 6, 3, 1],  # eyes - iris + pupil
        [1, 3, 7, 6, 7, 6, 3, 1],
        [1, 3, 3, 3, 3, 3, 3, 1],
        [1, 3, 3, 3, 3, 3, 3, 1],
        [1, 3, 3, 3, 3, 3, 3, 1],
        [1, 0, 1, 0, 1, 0, 1, 0],  # wavy bottom
    ], palette_type, desc)

_ghost("blinky",   "blinky", "Blinky - red ghost (Shadow)")
_ghost("pinky",    "pinky",  "Pinky - pink ghost (Speedy)")
_ghost("inky",     "inky",   "Inky - cyan ghost (Bashful)")
_ghost("clyde",    "clyde",  "Clyde - orange ghost (Pokey)")

_reg("pac_scared", "pacman", 8, [
    [0, 1, 1, 1, 1, 1, 1, 0],
    [1, 3, 3, 3, 3, 3, 3, 1],
    [1, 3, 6, 3, 3, 6, 3, 1],  # small scared dots
    [1, 3, 6, 3, 3, 6, 3, 1],
    [1, 3, 3, 3, 3, 3, 3, 1],
    [1, 3, 2, 3, 2, 3, 2, 1],  # jagged scared mouth
    [1, 3, 3, 3, 3, 3, 3, 1],
    [1, 0, 1, 0, 1, 0, 1, 0],
], "scared", "Scared ghost - power pellet active")

_reg("pac_ghost_eyes", "pacman", 8, [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 7, 6, 7, 6, 1, 0],  # just the eyes
    [0, 1, 7, 6, 7, 6, 1, 0],
    [0, 0, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
], "maze", "Ghost eyes - eaten ghost returning to base")

# Bonus fruits
_reg("pac_pellet", "pacman", 8, [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0],
    [0, 1, 4, 5, 4, 1, 0, 0],
    [0, 1, 5, 3, 3, 1, 0, 0],
    [0, 1, 4, 3, 3, 1, 0, 0],
    [0, 1, 3, 3, 3, 1, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
], "pellet", "Power pellet - big energizer dot")

_reg("pac_dot", "pacman", 8, [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
], "pellet", "Regular dot - maze pellet")

_reg("pac_cherry", "pacman", 8, [
    [0, 0, 1, 0, 0, 0, 1, 0],
    [0, 0, 1, 1, 0, 1, 1, 0],
    [0, 0, 0, 1, 1, 1, 0, 0],
    [0, 1, 3, 3, 1, 1, 3, 1],
    [1, 3, 4, 3, 3, 3, 4, 1],
    [1, 3, 5, 3, 3, 3, 5, 1],
    [1, 3, 3, 3, 3, 3, 3, 1],
    [0, 1, 1, 1, 0, 1, 1, 0],
], "cherry", "Cherry - 100 point bonus fruit")

_reg("pac_strawberry", "pacman", 8, [
    [0, 0, 0, 6, 6, 0, 0, 0],  # leaves
    [0, 0, 6, 6, 6, 6, 0, 0],
    [0, 1, 3, 3, 3, 3, 1, 0],  # berry body
    [1, 3, 4, 7, 3, 3, 3, 1],  # sheen + seeds
    [1, 3, 3, 7, 3, 7, 3, 1],  # seeds
    [1, 3, 3, 3, 7, 3, 3, 1],
    [0, 1, 3, 3, 3, 3, 1, 0],
    [0, 0, 1, 1, 1, 1, 0, 0],
], "blinky", "Strawberry - 300 point bonus fruit")

_reg("pac_orange_b", "pacman", 8, [
    [0, 0, 0, 6, 6, 0, 0, 0],  # leaves/stem
    [0, 0, 0, 6, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0, 0],
    [1, 3, 4, 5, 3, 3, 1, 0],
    [1, 3, 4, 3, 3, 3, 3, 1],
    [1, 3, 3, 3, 3, 3, 3, 1],
    [0, 1, 3, 3, 3, 3, 1, 0],
    [0, 0, 1, 1, 1, 1, 0, 0],
], "clyde", "Orange fruit - 500 point bonus")

_reg("pac_apple_b", "pacman", 8, [
    [0, 0, 0, 6, 0, 0, 0, 0],  # stem
    [0, 0, 6, 6, 6, 0, 0, 0],  # leaves
    [0, 1, 1, 1, 1, 1, 0, 0],
    [1, 3, 4, 5, 3, 3, 1, 0],
    [1, 3, 5, 3, 3, 3, 3, 1],
    [1, 3, 3, 3, 3, 3, 3, 1],
    [0, 1, 3, 3, 3, 2, 1, 0],
    [0, 0, 1, 1, 1, 1, 0, 0],
], "blinky", "Apple - 700 point bonus fruit")

_reg("pac_melon", "pacman", 8, [
    [0, 0, 0, 6, 6, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [1, 3, 4, 3, 3, 4, 3, 1],
    [1, 3, 3, 2, 2, 3, 3, 1],  # rind stripes
    [1, 3, 2, 3, 3, 2, 3, 1],
    [1, 3, 3, 2, 2, 3, 3, 1],
    [0, 1, 3, 3, 3, 3, 1, 0],
    [0, 0, 1, 1, 1, 1, 0, 0],
], "rupee_green", "Melon - 1000 point bonus fruit")


# ============================================================
#  ITEMS  (item_*)
# ============================================================

_reg("item_star", "item", 8, [
    [0, 0, 0, 7, 7, 0, 0, 0],
    [0, 0, 7, 6, 6, 7, 0, 0],
    [7, 7, 6, 5, 5, 6, 7, 7],
    [0, 7, 5, 6, 6, 5, 7, 0],
    [0, 7, 5, 6, 6, 5, 7, 0],
    [7, 7, 6, 5, 5, 6, 7, 7],
    [0, 0, 7, 6, 6, 7, 0, 0],
    [0, 0, 0, 7, 7, 0, 0, 0],
], "boss_gold", "Star - shine collectible")

_reg("item_coin", "item", 8, [
    [0, 0, 1, 1, 1, 1, 0, 0],
    [0, 1, 3, 5, 4, 3, 1, 0],
    [1, 3, 5, 5, 4, 4, 3, 1],
    [1, 3, 5, 6, 4, 3, 3, 1],
    [1, 3, 4, 3, 3, 3, 3, 1],
    [1, 3, 3, 3, 3, 3, 3, 1],
    [0, 1, 3, 3, 3, 3, 1, 0],
    [0, 0, 1, 1, 1, 1, 0, 0],
], "rupee_red", "Gold coin - generic collectible")

_reg("item_gem", "item", 8, [
    [0, 0, 1, 1, 1, 1, 0, 0],
    [0, 1, 5, 7, 4, 3, 1, 0],
    [1, 3, 7, 7, 5, 4, 3, 1],
    [1, 3, 5, 5, 4, 4, 3, 1],
    [0, 1, 4, 3, 3, 3, 1, 0],
    [0, 1, 3, 3, 3, 3, 1, 0],
    [0, 0, 1, 3, 3, 1, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0],
], "rupee_blue", "Gem - precious stone item")

_reg("item_potion", "item", 8, [
    [0, 0, 1, 1, 1, 0, 0, 0],  # stopper
    [0, 0, 1, 3, 1, 0, 0, 0],
    [0, 0, 1, 3, 1, 0, 0, 0],
    [0, 1, 1, 3, 1, 1, 0, 0],  # bottle shoulder
    [1, 3, 5, 7, 3, 3, 1, 0],  # liquid
    [1, 3, 5, 4, 3, 3, 1, 0],
    [1, 3, 3, 3, 3, 3, 1, 0],
    [0, 1, 1, 1, 1, 1, 0, 0],
], "fairy", "Potion - healing item")

_reg("item_mushroom", "item", 8, [
    [0, 0, 1, 1, 1, 1, 0, 0],
    [0, 1, 3, 5, 4, 3, 1, 0],  # cap top
    [1, 3, 5, 7, 3, 4, 3, 1],  # sheen + spots
    [1, 3, 3, 7, 3, 7, 3, 1],  # spots
    [1, 3, 3, 3, 3, 3, 3, 1],
    [0, 1, 5, 5, 5, 5, 1, 0],  # stem (lighter color)
    [0, 1, 5, 5, 5, 5, 1, 0],
    [0, 0, 1, 1, 1, 1, 0, 0],
], "blinky", "Mushroom - power-up item")

_reg("item_chest_open", "item", 8, [
    [0, 1, 7, 7, 7, 7, 1, 0],  # open lid with treasure glow
    [1, 3, 7, 6, 6, 7, 3, 1],  # glowing interior
    [1, 3, 6, 7, 7, 6, 3, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],  # box top edge
    [1, 2, 3, 3, 3, 3, 2, 1],
    [1, 2, 3, 3, 3, 3, 2, 1],
    [1, 2, 3, 3, 3, 3, 2, 1],
    [0, 1, 2, 2, 2, 2, 1, 0],
], "boss_gold", "Treasure chest - open state")


# ============================================================
#  TREES  (tree_*)  - hand-crafted for quality
#  Color index semantics for trees:
#  0=transparent, 1=bark dark, 2=bark mid, 3=bark light,
#  4=leaf shadow, 5=leaf mid, 6=leaf bright, 7=special (snow/blossom/bark marks)
# ============================================================

# Pine tree - summer (16x16, classic tiered silhouette)
_reg("tree_pine_summer", "nature", 16, [
    [0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0],  # tip highlight
    [0, 0, 0, 0, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0],  # tier 1 top
    [0, 0, 0, 0, 0, 4, 5, 6, 5, 4, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 4, 4, 5, 5, 5, 4, 4, 0, 0, 0, 0, 0],  # tier 1 base
    [0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0],  # trunk peek
    [0, 0, 0, 4, 4, 5, 5, 6, 5, 5, 4, 4, 0, 0, 0, 0],  # tier 2 top
    [0, 0, 4, 4, 5, 5, 5, 5, 5, 5, 5, 4, 4, 0, 0, 0],
    [0, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 0, 0],  # tier 2 base
    [0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0],  # trunk peek
    [0, 0, 4, 4, 4, 5, 5, 6, 5, 5, 4, 4, 4, 0, 0, 0],  # tier 3 top
    [0, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 0, 0],
    [4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 0],  # tier 3 base wide
    [0, 0, 0, 0, 0, 1, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0],  # trunk
    [0, 0, 0, 0, 0, 1, 2, 3, 2, 1, 0, 0, 0, 0, 0, 0],  # trunk lit
    [0, 0, 0, 0, 1, 2, 2, 3, 2, 2, 1, 0, 0, 0, 0, 0],  # trunk base
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], "pine_summer", "Pine tree - summer green (16x16)")

# Pine tree - winter (snow on branches)
_reg("tree_pine_winter", "nature", 16, [
    [0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0],  # snow tip
    [0, 0, 0, 0, 0, 0, 7, 7, 7, 0, 0, 0, 0, 0, 0, 0],  # snow cap
    [0, 0, 0, 0, 0, 4, 5, 7, 5, 4, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 7, 7, 5, 5, 5, 7, 7, 0, 0, 0, 0, 0],  # snow on needles
    [0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 4, 7, 7, 5, 7, 5, 7, 7, 4, 0, 0, 0, 0],  # snow tier
    [0, 0, 4, 4, 5, 7, 5, 5, 5, 7, 5, 4, 4, 0, 0, 0],
    [0, 7, 7, 7, 5, 5, 5, 5, 5, 5, 5, 7, 7, 7, 0, 0],  # heavy snow
    [0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 4, 7, 7, 7, 5, 7, 5, 7, 7, 7, 4, 0, 0, 0],
    [0, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5, 7, 7, 0, 0],
    [7, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 7, 7, 0],
    [0, 0, 0, 0, 0, 1, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 2, 3, 2, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 2, 2, 3, 2, 2, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], "pine_winter", "Pine tree - snow-laden winter (16x16)")

# Maple tree - summer (round canopy, 16x16)
_reg("tree_maple_summer", "nature", 16, [
    [0, 0, 0, 0, 4, 5, 5, 5, 5, 5, 4, 0, 0, 0, 0, 0],  # canopy top arc
    [0, 0, 0, 4, 5, 5, 6, 5, 6, 5, 5, 4, 0, 0, 0, 0],  # leaf highlight
    [0, 0, 4, 5, 5, 6, 5, 5, 5, 6, 5, 5, 4, 0, 0, 0],
    [0, 4, 5, 5, 6, 5, 5, 5, 5, 5, 6, 5, 5, 4, 0, 0],
    [4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 0],  # widest
    [4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 0],
    [0, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 0, 0],
    [0, 0, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 0, 0, 0],
    [0, 0, 0, 4, 4, 4, 2, 2, 2, 4, 4, 4, 0, 0, 0, 0],  # canopy base + trunk
    [0, 0, 0, 0, 0, 1, 2, 3, 2, 1, 0, 0, 0, 0, 0, 0],  # trunk
    [0, 0, 0, 0, 0, 1, 2, 3, 2, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 2, 2, 3, 2, 2, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 2, 3, 2, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], "maple_summer", "Maple tree - lush summer canopy (16x16)")

# Maple tree - autumn (same shape, warm palette)
_reg("tree_maple_autumn", "nature", 16, [
    [0, 0, 0, 0, 4, 5, 5, 5, 5, 5, 4, 0, 0, 0, 0, 0],
    [0, 0, 0, 4, 5, 5, 6, 5, 6, 5, 5, 4, 0, 0, 0, 0],
    [0, 0, 4, 5, 5, 6, 5, 5, 5, 6, 5, 5, 4, 0, 0, 0],
    [0, 4, 5, 6, 6, 5, 7, 5, 5, 5, 6, 5, 5, 4, 0, 0],  # autumn reds/golds
    [4, 5, 6, 5, 5, 7, 5, 5, 5, 5, 5, 7, 5, 5, 4, 0],
    [4, 5, 5, 6, 5, 5, 5, 5, 5, 5, 6, 5, 5, 5, 4, 0],
    [0, 4, 5, 5, 5, 5, 5, 7, 5, 5, 5, 5, 5, 4, 0, 0],
    [0, 0, 4, 5, 6, 5, 5, 5, 5, 5, 6, 5, 4, 0, 0, 0],
    [0, 0, 0, 4, 4, 4, 2, 2, 2, 4, 4, 4, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 2, 3, 2, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 2, 3, 2, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 2, 2, 3, 2, 2, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 2, 3, 2, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], "maple_autumn", "Maple tree - fiery autumn orange/red (16x16)")

# Maple tree - spring (pale green + pink blossoms)
_reg("tree_maple_spring", "nature", 16, [
    [0, 0, 0, 0, 7, 5, 7, 5, 7, 5, 7, 0, 0, 0, 0, 0],  # blossom tips
    [0, 0, 0, 7, 5, 7, 6, 5, 6, 7, 5, 7, 0, 0, 0, 0],
    [0, 0, 7, 5, 5, 6, 5, 5, 5, 6, 5, 5, 7, 0, 0, 0],
    [0, 7, 5, 5, 6, 5, 5, 7, 5, 5, 6, 5, 5, 7, 0, 0],
    [7, 5, 5, 7, 5, 5, 5, 5, 5, 5, 5, 5, 7, 5, 7, 0],  # blossoms + leaves
    [5, 5, 5, 5, 7, 5, 5, 5, 5, 5, 7, 5, 5, 5, 5, 0],
    [0, 5, 5, 5, 5, 5, 5, 7, 5, 5, 5, 5, 5, 5, 0, 0],
    [0, 0, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5, 7, 0, 0, 0],
    [0, 0, 0, 4, 4, 4, 2, 2, 2, 4, 4, 4, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 2, 3, 2, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 2, 3, 2, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 2, 2, 3, 2, 2, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 2, 3, 2, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], "maple_spring", "Maple tree - spring blossoms pink/white (16x16)")

# Birch tree - summer (tall, white bark, horizontal dark marks)
# Note: for birch, index 3=trunk light = WHITE/CREAM; index 7=bark marks
_reg("tree_birch_summer", "nature", 16, [
    [0, 0, 0, 0, 4, 5, 5, 5, 5, 5, 4, 0, 0, 0, 0, 0],  # small canopy
    [0, 0, 0, 4, 5, 5, 6, 5, 6, 5, 5, 4, 0, 0, 0, 0],
    [0, 0, 4, 5, 5, 6, 5, 5, 5, 6, 5, 5, 4, 0, 0, 0],
    [0, 0, 0, 4, 5, 5, 5, 5, 5, 5, 5, 4, 0, 0, 0, 0],
    [0, 0, 0, 0, 4, 5, 5, 5, 5, 5, 4, 0, 0, 0, 0, 0],  # canopy bottom
    [0, 0, 0, 0, 0, 1, 3, 3, 3, 1, 0, 0, 0, 0, 0, 0],  # white trunk
    [0, 0, 0, 0, 0, 1, 3, 7, 3, 1, 0, 0, 0, 0, 0, 0],  # bark mark
    [0, 0, 0, 0, 0, 1, 3, 3, 3, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 7, 3, 7, 1, 0, 0, 0, 0, 0, 0],  # bark marks (wide)
    [0, 0, 0, 0, 0, 1, 3, 3, 3, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 3, 7, 3, 1, 0, 0, 0, 0, 0, 0],  # bark mark
    [0, 0, 0, 0, 0, 1, 3, 3, 3, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 7, 3, 7, 1, 0, 0, 0, 0, 0, 0],  # bark marks
    [0, 0, 0, 0, 0, 1, 3, 3, 3, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 2, 3, 3, 3, 2, 1, 0, 0, 0, 0, 0],  # roots
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], "birch_summer", "Birch tree - white bark, summer leaves (16x16)")

# Birch tree - autumn (yellow-gold canopy)
_reg("tree_birch_autumn", "nature", 16, [
    [0, 0, 0, 0, 4, 5, 7, 5, 7, 5, 4, 0, 0, 0, 0, 0],
    [0, 0, 0, 4, 5, 7, 6, 5, 6, 7, 5, 4, 0, 0, 0, 0],
    [0, 0, 4, 5, 7, 6, 5, 5, 5, 6, 7, 5, 4, 0, 0, 0],
    [0, 0, 0, 4, 5, 5, 5, 5, 5, 5, 5, 4, 0, 0, 0, 0],
    [0, 0, 0, 0, 4, 5, 5, 5, 5, 5, 4, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 3, 3, 3, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 3, 7, 3, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 3, 3, 3, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 7, 3, 7, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 3, 3, 3, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 3, 7, 3, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 3, 3, 3, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 7, 3, 7, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 3, 3, 3, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 2, 3, 3, 3, 2, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], "birch_autumn", "Birch tree - golden autumn leaves (16x16)")

# Dead/bare winter tree (no leaves, dramatic silhouette)
_reg("tree_dead", "nature", 16, [
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # top bare branches
    [0, 0, 0, 0, 1, 2, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 2, 2, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 2, 1, 1, 2, 1, 0, 0, 0, 0, 0, 0],  # branches converge
    [0, 0, 0, 0, 0, 1, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 1, 2, 2, 1, 0, 1, 0, 0, 0, 0, 0],  # side branches
    [0, 1, 1, 0, 1, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0],  # main trunk
    [0, 0, 0, 0, 1, 2, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 2, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 2, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 2, 2, 3, 2, 2, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 2, 2, 3, 2, 2, 1, 0, 0, 0, 0, 0, 0],  # roots spread
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], "dead_tree", "Dead tree - bare winter silhouette (16x16)")

# Tree stump (8x8)
_reg("tree_stump", "nature", 8, [
    [0, 0, 1, 2, 2, 2, 1, 0],  # cross-section rings
    [0, 0, 1, 3, 2, 3, 1, 0],
    [0, 0, 1, 3, 3, 3, 1, 0],
    [0, 1, 2, 3, 3, 3, 2, 1],  # top edge
    [1, 2, 3, 3, 3, 3, 2, 1],  # body
    [1, 2, 3, 3, 3, 3, 2, 1],
    [1, 2, 2, 3, 3, 2, 2, 1],
    [0, 1, 1, 2, 2, 1, 1, 0],  # base
], "maple_summer", "Tree stump - chopped tree remains")

# Oak tree - 32x32 (larger, more detailed - for higher-res training)
_reg("tree_oak_summer", "nature", 32, [
    [0,0,0,0,0,0,0,0,0,4,5,5,5,5,5,5,5,5,4,0,0,0,0,0,0,0,0,0,0,0,0,0],  # canopy top
    [0,0,0,0,0,0,0,4,5,5,5,6,5,5,6,5,5,5,5,4,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,4,5,5,6,5,5,5,5,5,5,6,5,5,5,4,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,4,5,5,6,5,5,5,5,5,5,5,5,6,5,5,5,4,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,4,5,5,5,5,5,5,6,5,5,6,5,5,5,5,5,5,5,4,0,0,0,0,0,0,0,0,0],
    [0,0,0,4,5,5,5,6,5,5,5,5,5,5,5,5,5,5,5,6,5,5,5,4,0,0,0,0,0,0,0,0],
    [0,0,4,5,5,5,5,5,5,5,6,5,5,5,5,6,5,5,5,5,5,5,5,5,4,0,0,0,0,0,0,0],
    [0,4,5,5,5,5,5,6,5,5,5,5,5,5,5,5,5,5,6,5,5,5,5,5,5,4,0,0,0,0,0,0],
    [4,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,0,0,0,0,0],
    [4,5,5,5,5,5,6,5,5,5,5,5,5,5,5,5,5,5,5,5,6,5,5,5,5,5,4,0,0,0,0,0],
    [4,5,5,5,5,5,5,5,6,5,5,5,5,5,5,5,5,5,6,5,5,5,5,5,5,5,4,0,0,0,0,0],
    [0,4,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,0,0,0,0,0,0],
    [0,0,4,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,0,0,0,0,0,0,0],
    [0,0,0,4,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,0,0,0,0,0,0,0,0],
    [0,0,0,0,4,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,4,4,5,5,5,5,5,5,5,5,5,5,5,5,5,4,4,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,4,4,4,5,5,2,2,2,2,2,2,5,4,4,4,0,0,0,0,0,0,0,0,0,0,0],  # trunk emerges
    [0,0,0,0,0,0,0,0,0,1,2,2,3,3,2,2,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,1,2,3,3,3,3,2,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,1,2,2,3,3,2,2,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,1,2,3,3,3,3,2,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # trunk
    [0,0,0,0,0,0,0,0,0,1,2,2,3,3,2,2,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,1,2,3,3,3,3,2,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,2,2,2,3,3,2,2,2,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,2,2,2,3,3,2,2,2,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,2,2,2,2,3,3,2,2,2,2,2,1,0,0,0,0,0,0,0,0,0,0,0,0],  # roots spread
    [0,0,0,0,0,0,0,1,2,2,2,2,3,3,2,2,2,2,2,1,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,2,2,2,2,2,3,3,2,2,2,2,2,2,1,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
], "maple_summer", "Oak tree - large detailed summer oak (32x32)")


# ============================================================
#  Category listing
# ============================================================

CATEGORIES = ["galaga", "zelda", "pacman", "item", "nature"]


# ============================================================
#  Rendering engine (unchanged from original)
# ============================================================

@dataclass
class SpriteRenderer:
    """Renders sprite templates into RGBA numpy arrays."""

    def render(
        self,
        sprite_name: str,
        palette: ColorPalette,
        display_scale: int = 1,
        dither_method: str = "none",
        dither_intensity: float = 0.3,
    ) -> np.ndarray:
        spec = SPRITES[sprite_name]
        data = np.array(spec["data"], dtype=np.int32)
        size = spec["size"]

        rgba = np.zeros((size, size, 4), dtype=np.uint8)
        for y in range(size):
            for x in range(size):
                idx = int(data[y, x])
                if idx == 0:
                    rgba[y, x] = [0, 0, 0, 0]
                else:
                    palette_idx = min(idx, len(palette) - 1)
                    c = palette[palette_idx]
                    rgba[y, x] = [c.r, c.g, c.b, 255]

        if dither_method != "none":
            rgb = rgba[:, :, :3]
            alpha_mask = rgba[:, :, 3] > 0
            if alpha_mask.any():
                dithered = apply_dithering(rgb, palette, dither_method, dither_intensity)
                rgba[:, :, :3][alpha_mask] = dithered[alpha_mask]

        if display_scale > 1:
            rgba = np.repeat(np.repeat(rgba, display_scale, axis=0), display_scale, axis=1)

        return rgba

    def render_to_pil(self, sprite_name, palette, display_scale=4, **kw):
        arr = self.render(sprite_name, palette, display_scale, **kw)
        return Image.fromarray(arr, mode="RGBA")

    def render_to_bytes(self, sprite_name, palette, display_scale=1, **kw):
        img = self.render_to_pil(sprite_name, palette, display_scale, **kw)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


def generate_sprite_sheet(
    sprite_names: list[str],
    palette: ColorPalette,
    display_scale: int = 4,
    cols: int = 8,
    bg_color: tuple[int, int, int] = (15, 15, 25),
    padding: int = 2,
    dither_method: str = "none",
) -> Image.Image:
    renderer = SpriteRenderer()
    rendered = []
    max_size = 0

    for name in sprite_names:
        arr = renderer.render(name, palette, display_scale, dither_method)
        rendered.append((name, arr))
        max_size = max(max_size, arr.shape[0], arr.shape[1])

    rows = (len(rendered) + cols - 1) // cols
    cell = max_size + padding * 2
    sheet_w = cols * cell
    sheet_h = rows * cell

    sheet = np.zeros((sheet_h, sheet_w, 4), dtype=np.uint8)
    sheet[:, :, :3] = bg_color
    sheet[:, :, 3] = 255

    for i, (name, arr) in enumerate(rendered):
        row, col = divmod(i, cols)
        y_off = row * cell + padding + (max_size - arr.shape[0]) // 2
        x_off = col * cell + padding + (max_size - arr.shape[1]) // 2
        h, w = arr.shape[:2]
        alpha = arr[:, :, 3:4] / 255.0
        sheet[y_off:y_off+h, x_off:x_off+w, :3] = (
            arr[:, :, :3] * alpha +
            sheet[y_off:y_off+h, x_off:x_off+w, :3] * (1 - alpha)
        ).astype(np.uint8)

    return Image.fromarray(sheet, mode="RGBA")


def generate_training_batch(
    sprite_names: Optional[list[str]] = None,
    n_per_sprite: int = 8,
    target_size: int = 32,
    dither_methods: list[str] = ("none", "bayer4x4"),
    base_seed: int = 0,
    include_augmentations: bool = True,
) -> list[dict]:
    if sprite_names is None:
        sprite_names = list(SPRITES.keys())

    renderer = SpriteRenderer()
    samples = []
    rng = random.Random(base_seed)

    for sprite_name in sprite_names:
        spec = SPRITES[sprite_name]
        palettes = generate_palette_variations(spec["palette_type"], n_per_sprite, base_seed)

        for var_idx, palette in enumerate(palettes):
            dither = dither_methods[var_idx % len(dither_methods)]
            seed = base_seed + hash(sprite_name) + var_idx

            arr = renderer.render(sprite_name, palette,
                                  display_scale=1, dither_method=dither,
                                  dither_intensity=0.35)

            native_size = spec["size"]
            if native_size != target_size:
                img = Image.fromarray(arr, mode="RGBA")
                img = img.resize((target_size, target_size), Image.NEAREST)
                arr = np.array(img)

            buf = io.BytesIO()
            Image.fromarray(arr, mode="RGBA").save(buf, format="PNG")
            png_bytes = buf.getvalue()

            samples.append({
                "image_bytes": png_bytes,
                "sprite_name": sprite_name,
                "palette_name": palette.name,
                "category": spec["category"],
                "size": target_size,
                "seed": seed,
                "caption": f"pixel art {spec['category']} {spec.get('description', sprite_name)}",
            })

            if include_augmentations:
                arr_flip = arr[:, ::-1, :].copy()
                buf = io.BytesIO()
                Image.fromarray(arr_flip, mode="RGBA").save(buf, format="PNG")
                samples.append({
                    "image_bytes": buf.getvalue(),
                    "sprite_name": sprite_name + "_flipped",
                    "palette_name": palette.name,
                    "category": spec["category"],
                    "size": target_size,
                    "seed": seed + 10000,
                    "caption": f"pixel art {spec['category']} {spec.get('description', sprite_name)} flipped",
                })

    rng.shuffle(samples)
    return samples


def list_sprites(category: Optional[str] = None) -> list[str]:
    if category is None:
        return list(SPRITES.keys())
    return [k for k, v in SPRITES.items() if v["category"] == category]


def get_sprite_info(name: str) -> dict:
    if name not in SPRITES:
        raise KeyError(f"Unknown sprite: {name!r}. Available: {list(SPRITES.keys())}")
    return SPRITES[name]
