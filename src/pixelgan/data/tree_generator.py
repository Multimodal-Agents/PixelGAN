"""
Procedural pixel art tree generator.

Generates tree sprites procedurally using recursive branching + silhouette
filling algorithms. Outputs color-index arrays (0-7) matching the tree
palette semantic layout used by sprite_generator.py.

Tree palette index semantics:
  0 = transparent
  1 = bark dark  (deep shadow on trunk/branches)
  2 = bark mid   (main trunk/branch tone)
  3 = bark light (bark highlight)
  4 = leaf dark  (deep foliage shadow)
  5 = leaf mid   (main leaf / needle tone)
  6 = leaf light (leaf top / needle tip highlight)
  7 = special    (snow for winter, blossoms for spring, marks for birch)

Usage:
    from pixelgan.data.tree_generator import ProceduralTreeGenerator

    gen = ProceduralTreeGenerator()
    img = gen.generate("pine", size=16, seed=42, season="summer")
    img.save("pine_summer.png")

    # Training batch (returns list of dicts with image_bytes + caption)
    batch = generate_tree_batch("maple", size=32, n=50, base_seed=0)
"""

from __future__ import annotations

import io
import math
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image

from .color_palette import generate_tree_palette, TREE_PALETTE_CONFIGS


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

def _draw_line(canvas: np.ndarray, x0: int, y0: int, x1: int, y1: int,
               color: int, width: int = 1) -> None:
    """Bresenham's line algorithm with optional pixel width."""
    H, W = canvas.shape
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0

    while True:
        # Draw a square of `width` pixels centered at (x, y)
        r = width // 2
        for dy2 in range(-r, r + 1):
            for dx2 in range(-r, r + 1):
                nx, ny = x + dx2, y + dy2
                if 0 <= nx < W and 0 <= ny < H:
                    canvas[ny, nx] = color

        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


def _fill_ellipse(canvas: np.ndarray, cx: float, cy: float,
                  rx: float, ry: float, color_fn) -> None:
    """
    Fill an ellipse with colors from color_fn(relative_r) where
    relative_r is 0.0 (center) to 1.0 (edge).
    """
    H, W = canvas.shape
    x0, x1 = max(0, int(cx - rx) - 1), min(W - 1, int(cx + rx) + 1)
    y0, y1 = max(0, int(cy - ry) - 1), min(H - 1, int(cy + ry) + 1)

    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            # Normalized ellipse distance
            ex = (x - cx) / (rx + 0.5) if rx > 0 else 0.0
            ey = (y - cy) / (ry + 0.5) if ry > 0 else 0.0
            d = math.sqrt(ex * ex + ey * ey)
            if d <= 1.0:
                c = color_fn(d)
                if c != 0:
                    canvas[y, x] = c


def _leaf_color(d: float, rng: random.Random) -> int:
    """Map ellipse distance to leaf color index with subtle randomness."""
    noise = rng.random() * 0.12
    r = d + noise
    if r > 0.92:
        return 4  # dark edge
    elif r > 0.60:
        return 5  # mid leaf
    elif r > 0.30:
        return 6  # inner highlight
    else:
        if rng.random() < 0.25:
            return 7  # tiny texture detail (reused by caller for specific season)
        return 6


# ---------------------------------------------------------------------------
# Branch data structure
# ---------------------------------------------------------------------------

@dataclass
class Branch:
    x: float
    y: float
    angle: float   # degrees, 0=up, 90=right
    length: float
    width: int
    depth: int


# ---------------------------------------------------------------------------
# Main generator class
# ---------------------------------------------------------------------------

class ProceduralTreeGenerator:
    """
    Generate pixel art trees procedurally.

    Each tree type is grown via parameterized recursive branching.
    Output is an 8-index numpy canvas that can be colorized via
    generate_tree_palette().
    """

    # ------------------------------------------------------------------ presets
    # Params: (branch_angle, angle_var, length_shrink, width_shrink, n_branches,
    #          min_len, leaf_type, trunk_frac, has_leaves, branching_algo, extras)
    #
    # branching_algo options:
    #   "recursive"   — classic symmetric recursive branching (default)
    #   "monopodial"  — dominant main axis, shorter side branches (spruce)
    #   "weeping"     — branches droop progressively downward (willow)
    #   "herringbone" — alternating single-side branches along axis (fir)
    #   "spiral"      — 3-way symmetric with per-depth rotation (cherry)
    #   "multi_stem"  — multiple trunks from base (shrub/bush)
    #   "crown"       — radial fan from top of trunk (palm)
    PRESETS = {
        # ---- Original 6 types (updated with branching_algo) ----
        "pine": dict(
            branch_angle=42,     # steep upward branches
            angle_var=8,
            length_shrink=0.62,
            width_shrink=0.55,
            n_branches=2,
            min_len=1.5,
            leaf_type="needle",  # triangular spread
            trunk_frac=0.55,     # trunk takes 55% of height
            has_leaves=True,
            branching_algo="recursive",
        ),
        "maple": dict(
            branch_angle=50,
            angle_var=14,
            length_shrink=0.70,
            width_shrink=0.60,
            n_branches=2,
            min_len=2.0,
            leaf_type="round",
            trunk_frac=0.35,
            has_leaves=True,
            branching_algo="recursive",
        ),
        "birch": dict(
            branch_angle=38,
            angle_var=12,
            length_shrink=0.65,
            width_shrink=0.55,
            n_branches=2,
            min_len=1.8,
            leaf_type="sparse",
            trunk_frac=0.60,
            has_leaves=True,
            branching_algo="recursive",
        ),
        "oak": dict(
            branch_angle=55,
            angle_var=18,
            length_shrink=0.68,
            width_shrink=0.62,
            n_branches=3,
            min_len=2.5,
            leaf_type="round",
            trunk_frac=0.30,
            has_leaves=True,
            branching_algo="recursive",
        ),
        "dead": dict(
            branch_angle=48,
            angle_var=20,
            length_shrink=0.65,
            width_shrink=0.58,
            n_branches=2,
            min_len=1.5,
            leaf_type="none",
            trunk_frac=0.45,
            has_leaves=False,
            branching_algo="recursive",
        ),
        "generic": dict(
            branch_angle=45,
            angle_var=15,
            length_shrink=0.68,
            width_shrink=0.60,
            n_branches=2,
            min_len=2.0,
            leaf_type="round",
            trunk_frac=0.40,
            has_leaves=True,
            branching_algo="recursive",
        ),

        # ---- 9 new diverse tree types ----

        # Willow — weeping drooping branches, long hanging leaf strands
        "willow": dict(
            branch_angle=72,
            angle_var=14,
            length_shrink=0.70,
            width_shrink=0.58,
            n_branches=2,
            min_len=1.5,
            leaf_type="weeping",  # long drooping strands
            trunk_frac=0.45,
            has_leaves=True,
            branching_algo="weeping",
        ),

        # Spruce — monopodial dominant axis, Christmas-tree silhouette
        "spruce": dict(
            branch_angle=30,
            angle_var=6,
            length_shrink=0.58,
            width_shrink=0.52,
            n_branches=2,
            min_len=1.0,
            leaf_type="needle",
            trunk_frac=0.70,
            has_leaves=True,
            branching_algo="monopodial",
        ),

        # Cherry — 3-way spiral branching, wide spreading canopy
        "cherry": dict(
            branch_angle=52,
            angle_var=18,
            length_shrink=0.72,
            width_shrink=0.62,
            n_branches=3,
            min_len=2.0,
            leaf_type="round",
            trunk_frac=0.22,
            has_leaves=True,
            branching_algo="spiral",
        ),

        # Acacia — flat-top savanna silhouette, near-horizontal spread
        "acacia": dict(
            branch_angle=82,
            angle_var=8,
            length_shrink=0.72,
            width_shrink=0.65,
            n_branches=2,
            min_len=2.5,
            leaf_type="flat",   # wide horizontal fill
            trunk_frac=0.55,
            has_leaves=True,
            branching_algo="recursive",
        ),

        # Shrub — multi-stem bushy form, grows wide and low
        "shrub": dict(
            branch_angle=58,
            angle_var=22,
            length_shrink=0.68,
            width_shrink=0.65,
            n_branches=3,
            min_len=1.5,
            leaf_type="round",
            trunk_frac=0.12,
            has_leaves=True,
            branching_algo="multi_stem",
            n_stems=4,
        ),

        # Palm — narrow trunk, radiating crown fronds at top
        "palm": dict(
            branch_angle=72,
            angle_var=10,
            length_shrink=0.55,
            width_shrink=0.45,
            n_branches=7,
            min_len=2.0,
            leaf_type="frond",
            trunk_frac=0.72,
            has_leaves=True,
            branching_algo="crown",
        ),

        # Fir — herringbone alternating branches, regular spine pattern
        "fir": dict(
            branch_angle=38,
            angle_var=7,
            length_shrink=0.60,
            width_shrink=0.54,
            n_branches=2,
            min_len=1.2,
            leaf_type="needle",
            trunk_frac=0.62,
            has_leaves=True,
            branching_algo="herringbone",
        ),

        # Apple — round compact canopy with occasional fruit dots
        "apple": dict(
            branch_angle=48,
            angle_var=16,
            length_shrink=0.70,
            width_shrink=0.62,
            n_branches=2,
            min_len=2.2,
            leaf_type="dotted",  # round blobs + index-7 fruit dots
            trunk_frac=0.28,
            has_leaves=True,
            branching_algo="recursive",
        ),

        # Cypress — very narrow columnar form, dense compact needles
        "cypress": dict(
            branch_angle=20,
            angle_var=5,
            length_shrink=0.62,
            width_shrink=0.58,
            n_branches=2,
            min_len=1.5,
            leaf_type="dense_needle",
            trunk_frac=0.65,
            has_leaves=True,
            branching_algo="recursive",
        ),
    }

    # Map palette_type -> (tree_type, season)
    PALETTE_TO_TREE: dict[str, tuple[str, str]] = {
        # Original types
        "pine_summer":    ("pine",    "summer"),
        "pine_winter":    ("pine",    "winter"),
        "maple_summer":   ("maple",   "summer"),
        "maple_autumn":   ("maple",   "autumn"),
        "maple_spring":   ("maple",   "spring"),
        "birch_summer":   ("birch",   "summer"),
        "birch_autumn":   ("birch",   "autumn"),
        "dead_tree":      ("dead",    "bare"),
        # New types
        "willow_summer":  ("willow",  "summer"),
        "willow_spring":  ("willow",  "spring"),
        "spruce_summer":  ("spruce",  "summer"),
        "spruce_winter":  ("spruce",  "winter"),
        "cherry_spring":  ("cherry",  "spring"),
        "cherry_summer":  ("cherry",  "summer"),
        "acacia_summer":  ("acacia",  "summer"),
        "acacia_dry":     ("acacia",  "autumn"),
        "shrub_summer":   ("shrub",   "summer"),
        "shrub_autumn":   ("shrub",   "autumn"),
        "palm_tropical":  ("palm",    "summer"),
        "palm_dry":       ("palm",    "autumn"),
        "fir_summer":     ("fir",     "summer"),
        "fir_winter":     ("fir",     "winter"),
        "apple_summer":   ("apple",   "summer"),
        "apple_autumn":   ("apple",   "autumn"),
        "cypress_summer": ("cypress", "summer"),
        "cypress_winter": ("cypress", "winter"),
    }

    # ------------------------------------------------------------------ public

    def generate(
        self,
        tree_type: str,
        size: int = 16,
        seed: int = 0,
        season: str = "summer",
    ) -> np.ndarray:
        """
        Generate a tree as a (size, size) numpy array of color indices 0-7.

        Args:
            tree_type: "pine", "maple", "birch", "oak", "dead", "generic",
                       "willow", "spruce", "cherry", "acacia", "shrub",
                       "palm", "fir", "apple", "cypress"
            size: pixel grid size (8, 16, 32, 64)
            seed: random seed for reproducible variation
            season: "summer", "autumn", "winter", "spring", "bare"

        Returns:
            numpy array shape (size, size) dtype uint8, values 0-7
        """
        rng = random.Random(seed + hash(tree_type) + size)
        preset = self.PRESETS.get(tree_type, self.PRESETS["generic"])

        canvas = np.zeros((size, size), dtype=np.uint8)

        # --- Trunk ---
        cx = size // 2
        trunk_top = max(1, int(size * (1.0 - preset["trunk_frac"])))
        trunk_bot = size - 1
        trunk_w = max(1, size // 10)

        self._draw_trunk(canvas, cx, trunk_top, trunk_bot, trunk_w, rng,
                         is_birch=(tree_type == "birch"))

        # --- Branches ---
        initial_len = size * 0.30 + rng.uniform(-size * 0.03, size * 0.03)
        algo = preset.get("branching_algo", "recursive")
        leaf_tips: list[tuple[float, float, float]] = []  # (x, y, size)
        max_d = self._max_depth(size)

        if algo == "multi_stem":
            leaf_tips = self._do_multi_stem(canvas, preset, rng, size, cx, trunk_w)
        elif algo == "crown":
            leaf_tips = self._do_palm_crown(canvas, preset, rng, size, cx, trunk_w)
        else:
            root_branch = Branch(
                x=float(cx), y=float(trunk_top),
                angle=0.0,  # pointing up
                length=initial_len,
                width=max(1, trunk_w - 1),
                depth=0,
            )
            if algo == "monopodial":
                self._grow_monopodial(canvas, root_branch, preset, rng, leaf_tips, max_d)
            elif algo == "weeping":
                self._grow_weeping(canvas, root_branch, preset, rng, leaf_tips, max_d)
            elif algo == "herringbone":
                self._grow_herringbone(canvas, root_branch, preset, rng, leaf_tips, max_d)
            elif algo == "spiral":
                self._grow_spiral(canvas, root_branch, preset, rng, leaf_tips, max_d)
            else:  # recursive (default)
                self._grow_branch(canvas, root_branch, preset, rng, leaf_tips, max_d)

        # --- Leaves ---
        if preset["has_leaves"] and season != "bare":
            self._add_foliage(canvas, leaf_tips, preset, size, season, rng)

        # --- Season special effects ---
        self._apply_season(canvas, season, tree_type, rng, size)

        # --- Auto-fit: crop to content, re-center/ground in canvas ---
        # margin scales with canvas size: 0 at 8px, 1 at 16, 2 at 32, 4 at 64 …
        margin = size // 16  # 8→0, 16→1, 32→2, 64→4, 128→8
        canvas = self._fit_to_canvas(canvas, margin)

        return canvas

    def generate_from_palette_type(
        self,
        palette_type: str,
        size: int = 16,
        seed: int = 0,
    ) -> np.ndarray:
        """Convenience wrapper: derive tree_type + season from palette_type string."""
        tree_type, season = self.PALETTE_TO_TREE.get(
            palette_type, ("generic", "summer")
        )
        return self.generate(tree_type, size, seed, season)

    def render_to_image(
        self,
        canvas: np.ndarray,
        palette_type: str,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Colorize a canvas using the named tree palette, return RGBA PIL image."""
        cfg = TREE_PALETTE_CONFIGS.get(palette_type)
        if cfg:
            leaf_hue, bark_hue, special = cfg
            palette = generate_tree_palette(leaf_hue, bark_hue, special, seed=seed)
        else:
            palette = generate_tree_palette(120, 30, None, seed=seed)

        H, W = canvas.shape
        rgba = np.zeros((H, W, 4), dtype=np.uint8)

        for idx in range(8):
            mask = canvas == idx
            if idx == 0:
                rgba[mask] = [0, 0, 0, 0]
            else:
                c = palette[idx]
                rgba[mask] = [c.r, c.g, c.b, 255]

        return Image.fromarray(rgba, mode="RGBA")

    # ------------------------------------------------------------------ internals

    def _max_depth(self, size: int) -> int:
        if size <= 8:
            return 2
        elif size <= 16:
            return 3
        elif size <= 32:
            return 4
        elif size <= 64:
            return 6
        return 7

    def _fit_to_canvas(self, canvas: np.ndarray, margin: int) -> np.ndarray:
        """
        Auto-fit the generated tree into a consistently framed square canvas:
          - Find bounding box of all drawn (non-zero) pixels
          - Scale content *down* only if it overflows (size - 2*margin)
          - Ground the tree at the bottom: roots touch (H - 1 - margin)
          - Center horizontally
        This gives every tree type the same clean square framing regardless
        of branching algorithm or canvas size.
        """
        H, W = canvas.shape
        rows = np.any(canvas > 0, axis=1)
        cols = np.any(canvas > 0, axis=0)
        if not rows.any():
            return canvas

        rmin = int(np.argmax(rows))
        rmax = int(H - 1 - np.argmax(rows[::-1]))
        cmin = int(np.argmax(cols))
        cmax = int(W - 1 - np.argmax(cols[::-1]))

        content = canvas[rmin:rmax + 1, cmin:cmax + 1]
        ch, cw = content.shape

        # Available space within margins
        avail = max(1, H - 2 * margin)

        # Scale down only if content overflows (never upscale — keeps pixel art crisp)
        scale = min(1.0, avail / max(ch, cw, 1))
        if scale < 1.0:
            new_h = max(1, int(ch * scale))
            new_w = max(1, int(cw * scale))
            iy = (np.arange(new_h) * ch / new_h).astype(int)
            ix = (np.arange(new_w) * cw / new_w).astype(int)
            content = content[np.ix_(iy, ix)]
            ch, cw = content.shape

        out = np.zeros((H, W), dtype=np.uint8)

        # Ground at bottom (tree roots at bottom margin row)
        y1 = H - margin          # exclusive end row
        y0 = max(margin, y1 - ch)  # start row
        placed_h = y1 - y0

        # Center horizontally
        x0 = max(0, (W - cw) // 2)
        x1 = min(W, x0 + cw)
        placed_w = x1 - x0

        out[y0:y1, x0:x1] = content[ch - placed_h:, :placed_w]
        return out

    def _draw_trunk(
        self, canvas: np.ndarray,
        cx: int, top: int, bot: int, w: int,
        rng: random.Random,
        is_birch: bool = False,
    ) -> None:
        """Draw main trunk with bark shading."""
        H, W = canvas.shape
        for y in range(top, bot + 1):
            # Trunk widens slightly at base
            taper = 1.0 + 0.3 * ((y - top) / max(1, bot - top))
            hw = max(0, int(w * taper * 0.5))
            for dx in range(-hw - 1, hw + 2):
                x = cx + dx
                if 0 <= x < W:
                    # Shading: dark sides, lighter center
                    rel = abs(dx) / (hw + 1) if hw > 0 else 0.0
                    if rel > 0.75:
                        canvas[y, x] = 1  # dark edge
                    elif rel > 0.35:
                        canvas[y, x] = 2  # mid bark
                    else:
                        canvas[y, x] = 3  # light center

        # Birch horizontal marks (index 7 = special = dark marks)
        if is_birch and bot - top > 3:
            n_marks = max(1, (bot - top) // 4)
            for i in range(n_marks):
                my = top + (i + 1) * (bot - top) // (n_marks + 1)
                mark_w = max(1, w // 2 + rng.randint(-1, 1))
                for dx in range(-mark_w, mark_w + 1):
                    x = cx + dx
                    if 0 <= x < W and 0 <= my < H:
                        if canvas[my, x] != 0:
                            canvas[my, x] = 7  # birch mark

    def _grow_branch(
        self,
        canvas: np.ndarray,
        branch: Branch,
        preset: dict,
        rng: random.Random,
        leaf_tips: list,
        max_depth: int,
    ) -> None:
        """Recursively grow branches, collect leaf tip positions."""
        H, W = canvas.shape

        # End point of this branch segment
        rad = math.radians(branch.angle - 90)  # 0=up in screen coords
        ex = branch.x + branch.length * math.cos(rad)
        ey = branch.y + branch.length * math.sin(rad)

        # Draw segment
        color = 2 if branch.width > 1 else 1
        _draw_line(canvas,
                   int(round(branch.x)), int(round(branch.y)),
                   int(round(ex)), int(round(ey)),
                   color, branch.width)

        if branch.length < preset["min_len"] or branch.depth >= max_depth:
            # Terminal: this is a leaf attachment point
            leaf_tips.append((ex, ey, branch.length * 1.2))
            return

        # Sprout child branches
        n = preset["n_branches"]
        # Spread angles evenly around branch direction, biased upward
        base_spread = preset["branch_angle"]
        for i in range(n):
            if n == 1:
                offset = 0.0
            else:
                offset = (i / (n - 1) - 0.5) * 2 * base_spread
            child_angle = branch.angle + offset + rng.uniform(
                -preset["angle_var"], preset["angle_var"]
            )
            child_len = branch.length * (
                preset["length_shrink"] + rng.uniform(-0.05, 0.05)
            )
            child_w = max(1, int(branch.width * preset["width_shrink"]))

            child = Branch(
                x=ex, y=ey,
                angle=child_angle,
                length=child_len,
                width=child_w,
                depth=branch.depth + 1,
            )
            self._grow_branch(canvas, child, preset, rng, leaf_tips, max_depth)

    # ---------------------------------------------------------------- new algos

    def _grow_monopodial(
        self,
        canvas: np.ndarray,
        branch: Branch,
        preset: dict,
        rng: random.Random,
        leaf_tips: list,
        max_depth: int,
    ) -> None:
        """
        Monopodial growth: dominant central axis continues straight up with
        shorter side branches at each node. Produces spruce/Christmas-tree form.
        """
        H, W = canvas.shape
        rad = math.radians(branch.angle - 90)
        ex = branch.x + branch.length * math.cos(rad)
        ey = branch.y + branch.length * math.sin(rad)

        color = 2 if branch.width > 1 else 1
        _draw_line(canvas,
                   int(round(branch.x)), int(round(branch.y)),
                   int(round(ex)), int(round(ey)),
                   color, branch.width)

        if branch.length < preset["min_len"] or branch.depth >= max_depth:
            leaf_tips.append((ex, ey, branch.length * 1.2))
            return

        # Main axis continues (slight wobble maintained upward)
        main_len = branch.length * (preset["length_shrink"] + rng.uniform(-0.03, 0.03))
        main_child = Branch(
            x=ex, y=ey,
            angle=branch.angle + rng.uniform(-5, 5),
            length=main_len,
            width=max(1, int(branch.width * preset["width_shrink"])),
            depth=branch.depth + 1,
        )
        self._grow_monopodial(canvas, main_child, preset, rng, leaf_tips, max_depth)

        # Side branches: shorter, near-perpendicular, limited recursion depth
        side_factor = max(0.18, 0.62 - branch.depth * 0.07)  # shorter near top
        for sign in (-1, 1):
            side_angle = branch.angle + sign * (72 + rng.uniform(-10, 10))
            side_len = branch.length * side_factor
            side_branch = Branch(
                x=ex, y=ey,
                angle=side_angle,
                length=side_len,
                width=max(1, branch.width - 1),
                depth=branch.depth + 1,
            )
            self._grow_branch(canvas, side_branch, preset, rng, leaf_tips,
                               max_depth=min(max_depth, branch.depth + 3))

    def _grow_weeping(
        self,
        canvas: np.ndarray,
        branch: Branch,
        preset: dict,
        rng: random.Random,
        leaf_tips: list,
        max_depth: int,
    ) -> None:
        """
        Weeping growth: branches droop progressively downward with depth.
        Willow style — starts spreading outward, gravity curves toward ground.
        """
        H, W = canvas.shape
        rad = math.radians(branch.angle - 90)
        ex = branch.x + branch.length * math.cos(rad)
        ey = branch.y + branch.length * math.sin(rad)

        color = 2 if branch.width > 1 else 1
        _draw_line(canvas,
                   int(round(branch.x)), int(round(branch.y)),
                   int(round(ex)), int(round(ey)),
                   color, branch.width)

        if branch.length < preset["min_len"] or branch.depth >= max_depth:
            leaf_tips.append((ex, ey, branch.length * 2.0))  # extra long for strands
            return

        n = preset["n_branches"]
        base_spread = preset["branch_angle"]
        droop_per_depth = 22.0  # degrees toward downward per recursion level

        for i in range(n):
            offset = (i / (n - 1) - 0.5) * 2 * base_spread if n > 1 else 0.0
            droop = branch.depth * droop_per_depth
            child_angle = branch.angle + offset + droop + rng.uniform(
                -preset["angle_var"], preset["angle_var"]
            )
            child_len = branch.length * (preset["length_shrink"] + rng.uniform(-0.05, 0.05))
            child_w = max(1, int(branch.width * preset["width_shrink"]))
            child = Branch(
                x=ex, y=ey,
                angle=child_angle,
                length=child_len,
                width=child_w,
                depth=branch.depth + 1,
            )
            self._grow_weeping(canvas, child, preset, rng, leaf_tips, max_depth)

    def _grow_herringbone(
        self,
        canvas: np.ndarray,
        branch: Branch,
        preset: dict,
        rng: random.Random,
        leaf_tips: list,
        max_depth: int,
    ) -> None:
        """
        Herringbone growth: alternating single-side branches along main axis.
        Fir/alternate-leaf style — regular, spine-like branching pattern.
        """
        H, W = canvas.shape
        rad = math.radians(branch.angle - 90)
        ex = branch.x + branch.length * math.cos(rad)
        ey = branch.y + branch.length * math.sin(rad)

        color = 2 if branch.width > 1 else 1
        _draw_line(canvas,
                   int(round(branch.x)), int(round(branch.y)),
                   int(round(ex)), int(round(ey)),
                   color, branch.width)

        if branch.length < preset["min_len"] or branch.depth >= max_depth:
            leaf_tips.append((ex, ey, branch.length * 1.2))
            return

        # Main axis continues
        main_len = branch.length * (preset["length_shrink"] + rng.uniform(-0.03, 0.03))
        main_child = Branch(
            x=ex, y=ey,
            angle=branch.angle + rng.uniform(-4, 4),
            length=main_len,
            width=max(1, int(branch.width * preset["width_shrink"])),
            depth=branch.depth + 1,
        )
        self._grow_herringbone(canvas, main_child, preset, rng, leaf_tips, max_depth)

        # Single alternating side branch (left on even depth, right on odd)
        side_sign = 1 if (branch.depth % 2 == 0) else -1
        side_angle = branch.angle + side_sign * (58 + rng.uniform(-8, 8))
        side_len = branch.length * (preset["length_shrink"] * 0.75 + rng.uniform(-0.04, 0.04))
        side_branch = Branch(
            x=ex, y=ey,
            angle=side_angle,
            length=side_len,
            width=max(1, branch.width - 1),
            depth=branch.depth + 1,
        )
        self._grow_branch(canvas, side_branch, preset, rng, leaf_tips,
                          max_depth=min(max_depth, branch.depth + 3))

    def _grow_spiral(
        self,
        canvas: np.ndarray,
        branch: Branch,
        preset: dict,
        rng: random.Random,
        leaf_tips: list,
        max_depth: int,
    ) -> None:
        """
        Spiral growth: n-way branching with per-depth rotational offset.
        Cherry/ornamental style — spreading full canopy with rotational variety.
        """
        H, W = canvas.shape
        rad = math.radians(branch.angle - 90)
        ex = branch.x + branch.length * math.cos(rad)
        ey = branch.y + branch.length * math.sin(rad)

        color = 2 if branch.width > 1 else 1
        _draw_line(canvas,
                   int(round(branch.x)), int(round(branch.y)),
                   int(round(ex)), int(round(ey)),
                   color, branch.width)

        if branch.length < preset["min_len"] or branch.depth >= max_depth:
            leaf_tips.append((ex, ey, branch.length * 1.4))
            return

        n = preset.get("n_branches", 3)
        base_spread = preset["branch_angle"]
        # Rotate the whole fan slightly each depth level for spiral variety
        rotation_offset = (branch.depth % 4) * 12.0

        for i in range(n):
            fan_offset = (i / (n - 1) - 0.5) * 2 * base_spread if n > 1 else 0.0
            child_angle = branch.angle + fan_offset + rotation_offset + rng.uniform(
                -preset["angle_var"], preset["angle_var"]
            )
            child_len = branch.length * (preset["length_shrink"] + rng.uniform(-0.05, 0.05))
            child_w = max(1, int(branch.width * preset["width_shrink"]))
            child = Branch(
                x=ex, y=ey,
                angle=child_angle,
                length=child_len,
                width=child_w,
                depth=branch.depth + 1,
            )
            self._grow_spiral(canvas, child, preset, rng, leaf_tips, max_depth)

    def _do_multi_stem(
        self,
        canvas: np.ndarray,
        preset: dict,
        rng: random.Random,
        size: int,
        cx: int,
        trunk_w: int,
    ) -> list:
        """
        Multi-stem growth: several trunks diverge from base (shrub/bush style).
        Returns leaf_tips list.
        """
        leaf_tips: list[tuple[float, float, float]] = []
        n_stems = preset.get("n_stems", 4)
        base_y = size - 1
        mid_y = max(1, int(size * (1.0 - preset["trunk_frac"] * 1.5)))
        max_depth = self._max_depth(size)

        for s in range(n_stems):
            t = s / (n_stems - 1) if n_stems > 1 else 0.5
            lean = (t - 0.5) * 60.0 + rng.uniform(-10, 10)  # lean from -30 to +30 deg
            stem_cx = int(cx + (t - 0.5) * size * 0.28)
            stem_top_y = int(mid_y + rng.uniform(-size * 0.06, size * 0.06))
            stem_w = max(1, trunk_w - 1)
            # Draw this stem's trunk
            self._draw_trunk(canvas, stem_cx, stem_top_y, base_y, stem_w, rng)
            # Branch from stem top
            stem_root = Branch(
                x=float(stem_cx), y=float(stem_top_y),
                angle=lean,
                length=size * (0.22 + rng.uniform(-0.03, 0.04)),
                width=max(1, stem_w - 1),
                depth=0,
            )
            self._grow_branch(canvas, stem_root, preset, rng, leaf_tips, max_depth)

        return leaf_tips

    def _do_palm_crown(
        self,
        canvas: np.ndarray,
        preset: dict,
        rng: random.Random,
        size: int,
        cx: int,
        trunk_w: int,
    ) -> list:
        """
        Palm crown: narrow trunk with radiating fronds fanning from the top.
        Returns leaf_tips list.
        """
        leaf_tips: list[tuple[float, float, float]] = []
        trunk_top = max(1, int(size * (1.0 - preset["trunk_frac"])))
        trunk_bot = size - 1

        # Draw the straight narrow trunk
        self._draw_trunk(canvas, cx, trunk_top, trunk_bot, trunk_w, rng)

        # Crown fronds radiate upward in a fan from the crown point
        n_fronds = preset.get("n_branches", 7)
        frond_len = size * 0.32 + rng.uniform(-size * 0.04, size * 0.04)
        crown_y = trunk_top

        for i in range(n_fronds):
            t = i / (n_fronds - 1) if n_fronds > 1 else 0.5
            # Spread from -80° to +80° relative to straight up
            angle = -80 + t * 160 + rng.uniform(-6, 6)
            # Outer fronds droop more (add downward bias proportional to spread)
            droop = abs(angle) * 0.35
            end_angle = angle + droop
            rad = math.radians(end_angle - 90)
            ex = cx + frond_len * math.cos(rad)
            ey = crown_y + frond_len * math.sin(rad)
            _draw_line(canvas, cx, crown_y, int(round(ex)), int(round(ey)), 1, 1)
            leaf_tips.append((ex, ey, frond_len * 0.85))

        return leaf_tips

    def _add_foliage(
        self,
        canvas: np.ndarray,
        leaf_tips: list,
        preset: dict,
        size: int,
        season: str,
        rng: random.Random,
    ) -> None:
        """Draw leaf/needle clusters at branch tip positions."""
        if not leaf_tips:
            return

        leaf_type = preset["leaf_type"]

        for tx, ty, tip_len in leaf_tips:
            r = max(1.0, tip_len * 0.8)

            if leaf_type == "needle":
                # Pine: triangular needle cluster (wider than tall)
                rx, ry = r * 1.3, r * 0.7
                _fill_ellipse(
                    canvas, tx, ty, rx, ry,
                    lambda d, rng=rng: _leaf_color(d, rng) if d < 1.0 else 0,
                )
            elif leaf_type == "round":
                # Maple / Oak: roughly circular blob
                jx = rng.uniform(-r * 0.2, r * 0.2)
                jy = rng.uniform(-r * 0.2, r * 0.2)
                _fill_ellipse(
                    canvas, tx + jx, ty + jy, r, r,
                    lambda d, rng=rng: _leaf_color(d, rng) if d < 1.0 else 0,
                )
            elif leaf_type == "sparse":
                # Birch: small sparse clusters
                for _ in range(max(1, int(r))):
                    sx = tx + rng.uniform(-r * 0.8, r * 0.8)
                    sy = ty + rng.uniform(-r * 0.4, r * 0.4)
                    sr = max(0.8, r * 0.4)
                    _fill_ellipse(
                        canvas, sx, sy, sr, sr,
                        lambda d, rng=rng: _leaf_color(d, rng) if d < 1.0 else 0,
                    )

            elif leaf_type == "weeping":
                # Willow: thin hanging strands that droop straight down
                H, W = canvas.shape
                n_strands = max(3, int(r * 2.5))
                for _ in range(n_strands):
                    sx = tx + rng.uniform(-r * 0.6, r * 0.6)
                    sy = ty + rng.uniform(-r * 0.25, r * 0.15)
                    strand_len = r * (0.9 + rng.uniform(0, 1.2))
                    for step in range(int(strand_len) + 1):
                        t = step / max(1, strand_len)
                        px = int(round(sx + rng.uniform(-0.4, 0.4)))
                        py = int(round(sy + strand_len * t))
                        if 0 <= px < W and 0 <= py < H:
                            c = 4 if t > 0.70 else (5 if t > 0.35 else 6)
                            canvas[py, px] = c

            elif leaf_type == "frond":
                # Palm: elongated leaflet blob along the frond direction
                rx, ry = r * 1.8, max(0.7, r * 0.45)
                _fill_ellipse(
                    canvas, tx, ty, rx, ry,
                    lambda d, rng=rng: _leaf_color(d, rng) if d < 1.0 else 0,
                )

            elif leaf_type == "flat":
                # Acacia: wide flat horizontal canopy layer
                rx, ry = r * 2.4, max(0.7, r * 0.38)
                _fill_ellipse(
                    canvas, tx, ty, rx, ry,
                    lambda d, rng=rng: _leaf_color(d, rng) if d < 1.0 else 0,
                )

            elif leaf_type == "dotted":
                # Apple: round canopy with scattered index-7 fruit dots
                jx = rng.uniform(-r * 0.15, r * 0.15)
                jy = rng.uniform(-r * 0.15, r * 0.15)
                def _apple_col(d, rng=rng):
                    c = _leaf_color(d, rng)
                    if c in (5, 6) and rng.random() < 0.12:
                        return 7  # fruit dot
                    return c
                _fill_ellipse(canvas, tx + jx, ty + jy, r, r, _apple_col)

            elif leaf_type == "dense_needle":
                # Cypress: tight compact needle clusters for columnar form
                rx, ry = r * 0.75, r * 0.65
                _fill_ellipse(
                    canvas, tx, ty, rx, ry,
                    lambda d, rng=rng: _leaf_color(d, rng) if d < 1.0 else 0,
                )

    def _apply_season(
        self,
        canvas: np.ndarray,
        season: str,
        tree_type: str,
        rng: random.Random,
        size: int,
    ) -> None:
        """Apply season-specific pixel overrides to existing canvas."""
        if season == "winter":
            # Snow: replace top-facing leaf pixels with index 7 (snow color)
            H, W = canvas.shape
            for y in range(H - 1):
                for x in range(W):
                    if canvas[y, x] in (4, 5, 6):
                        # Check if exposed (pixel above is transparent or bark)
                        above = canvas[y - 1, x] if y > 0 else 0
                        if above == 0 and rng.random() < 0.55:
                            canvas[y, x] = 7

        elif season == "spring" and tree_type in (
            "maple", "birch", "generic", "willow", "cherry", "apple"
        ):
            # Blossoms: randomly replace some leaf pixels with index 7 (blossom color)
            H, W = canvas.shape
            for y in range(H):
                for x in range(W):
                    if canvas[y, x] in (5, 6) and rng.random() < 0.28:
                        canvas[y, x] = 7

        elif season == "autumn" and tree_type in (
            "maple", "birch", "generic", "cherry", "acacia", "apple", "willow", "shrub"
        ):
            # Autumn: shift some mid-greens to index 7 for extra variety
            # (palette handles the orange hue; index 7 adds golden flecks)
            H, W = canvas.shape
            for y in range(H):
                for x in range(W):
                    if canvas[y, x] == 6 and rng.random() < 0.20:
                        canvas[y, x] = 7


# ---------------------------------------------------------------------------
# Batch generation helpers
# ---------------------------------------------------------------------------

# Season captions per tree type
_CAPTIONS: dict[str, dict[str, str]] = {
    "pine": {
        "summer": "pixel art pine tree green needles summer forest",
        "winter": "pixel art pine tree snow covered winter white branches",
        "autumn": "pixel art pine tree autumn deep green conifer",
        "spring": "pixel art pine tree spring forest green",
        "bare":   "pixel art pine tree dark evergreen",
    },
    "maple": {
        "summer": "pixel art maple tree green canopy summer leaves",
        "autumn": "pixel art maple tree autumn orange red fall leaves",
        "spring": "pixel art maple tree spring pink blossoms flowers",
        "winter": "pixel art maple tree winter bare snow",
        "bare":   "pixel art maple tree bare branches winter",
    },
    "birch": {
        "summer": "pixel art birch tree white bark summer green leaves",
        "autumn": "pixel art birch tree autumn golden yellow leaves white bark",
        "spring": "pixel art birch tree spring light green new leaves",
        "winter": "pixel art birch tree winter white trunk snow",
        "bare":   "pixel art birch tree white bark bare slender",
    },
    "oak": {
        "summer": "pixel art oak tree large green canopy summer",
        "autumn": "pixel art oak tree autumn red orange leaves",
        "spring": "pixel art oak tree spring fresh green leaves",
        "winter": "pixel art oak tree winter bare branches",
        "bare":   "pixel art oak tree bare gnarly branches",
    },
    "dead": {
        "bare":   "pixel art dead tree bare winter silhouette dark branches",
        "summer": "pixel art dead tree bare grey haunted",
        "winter": "pixel art dead tree winter bare snow silhouette",
        "autumn": "pixel art dead tree autumn bare spooky",
        "spring": "pixel art dead tree bare weathered",
    },
    "generic": {
        "summer": "pixel art tree green leaves summer",
        "autumn": "pixel art tree autumn orange leaves fall",
        "winter": "pixel art tree winter bare branches",
        "spring": "pixel art tree spring blossoms",
        "bare":   "pixel art tree bare branches",
    },
    # --- new types ---
    "willow": {
        "summer": "pixel art willow tree drooping branches green summer",
        "spring": "pixel art willow tree pale green spring weeping branches",
        "autumn": "pixel art willow tree golden autumn weeping",
        "winter": "pixel art willow tree bare winter weeping branches",
        "bare":   "pixel art willow tree bare weeping silhouette",
    },
    "spruce": {
        "summer": "pixel art spruce tree dark green conifer summer",
        "winter": "pixel art spruce tree snow covered christmas tree winter",
        "autumn": "pixel art spruce tree autumn evergreen",
        "spring": "pixel art spruce tree spring bright green",
        "bare":   "pixel art spruce tree evergreen silhouette",
    },
    "cherry": {
        "spring": "pixel art cherry blossom tree pink flowers spring sakura",
        "summer": "pixel art cherry tree green canopy summer",
        "autumn": "pixel art cherry tree autumn golden red leaves",
        "winter": "pixel art cherry tree winter bare branches",
        "bare":   "pixel art cherry tree bare branches winter",
    },
    "acacia": {
        "summer": "pixel art acacia tree flat canopy savanna summer",
        "autumn": "pixel art acacia tree dry season golden canopy",
        "winter": "pixel art acacia tree dry bare flat top",
        "spring": "pixel art acacia tree spring fresh green flat top",
        "bare":   "pixel art acacia tree bare flat top silhouette",
    },
    "shrub": {
        "summer": "pixel art shrub bush green summer dense foliage",
        "autumn": "pixel art shrub autumn orange yellow leaves",
        "winter": "pixel art shrub bare winter branches",
        "spring": "pixel art shrub spring fresh green",
        "bare":   "pixel art shrub bare winter bush",
    },
    "palm": {
        "summer": "pixel art palm tree tropical island beach summer",
        "autumn": "pixel art palm tree dry season",
        "winter": "pixel art palm tree winter tropical",
        "spring": "pixel art palm tree spring tropical fronds",
        "bare":   "pixel art palm tree bare trunk fronds",
    },
    "fir": {
        "summer": "pixel art fir tree dark green conifer summer",
        "winter": "pixel art fir tree snow covered winter conifer",
        "autumn": "pixel art fir tree autumn deep green fir",
        "spring": "pixel art fir tree spring bright green tips",
        "bare":   "pixel art fir tree bare winter branches",
    },
    "apple": {
        "summer": "pixel art apple tree green summer fruit growing",
        "autumn": "pixel art apple tree autumn harvest red apples fruit",
        "spring": "pixel art apple tree spring white blossom flowers",
        "winter": "pixel art apple tree winter bare branches",
        "bare":   "pixel art apple tree bare winter branches",
    },
    "cypress": {
        "summer": "pixel art cypress tree tall narrow columnar summer",
        "winter": "pixel art cypress tree snow narrow columnar winter",
        "autumn": "pixel art cypress tree autumn dark evergreen",
        "spring": "pixel art cypress tree spring green columnar",
        "bare":   "pixel art cypress tree columnar silhouette",
    },
}

# Map palette_type -> (tree_type, season) for convenience
_PALETTE_TO_TREE: dict[str, tuple[str, str]] = {
    # Original types
    "pine_summer":    ("pine",    "summer"),
    "pine_winter":    ("pine",    "winter"),
    "maple_summer":   ("maple",   "summer"),
    "maple_autumn":   ("maple",   "autumn"),
    "maple_spring":   ("maple",   "spring"),
    "birch_summer":   ("birch",   "summer"),
    "birch_autumn":   ("birch",   "autumn"),
    "dead_tree":      ("dead",    "bare"),
    # New types
    "willow_summer":  ("willow",  "summer"),
    "willow_spring":  ("willow",  "spring"),
    "spruce_summer":  ("spruce",  "summer"),
    "spruce_winter":  ("spruce",  "winter"),
    "cherry_spring":  ("cherry",  "spring"),
    "cherry_summer":  ("cherry",  "summer"),
    "acacia_summer":  ("acacia",  "summer"),
    "acacia_dry":     ("acacia",  "autumn"),
    "shrub_summer":   ("shrub",   "summer"),
    "shrub_autumn":   ("shrub",   "autumn"),
    "palm_tropical":  ("palm",    "summer"),
    "palm_dry":       ("palm",    "autumn"),
    "fir_summer":     ("fir",     "summer"),
    "fir_winter":     ("fir",     "winter"),
    "apple_summer":   ("apple",   "summer"),
    "apple_autumn":   ("apple",   "autumn"),
    "cypress_summer": ("cypress", "summer"),
    "cypress_winter": ("cypress", "winter"),
}


# All tree type names exported for use by external scripts
TREE_TYPES: list[str] = list(ProceduralTreeGenerator.PRESETS.keys())


def generate_tree_batch(
    palette_type: str,
    size: int = 16,
    n: int = 20,
    base_seed: int = 0,
    include_flipped: bool = True,
    display_scale: int = 1,
) -> list[dict]:
    """
    Generate N procedural tree images as training samples.

    Returns a list of dicts with keys:
      image_bytes, sprite_name, palette_name, category, size, seed, caption

    Compatible with generate_training_batch() output format.
    """
    tree_type, season = _PALETTE_TO_TREE.get(palette_type, ("generic", "summer"))
    gen = ProceduralTreeGenerator()
    samples = []

    for i in range(n):
        seed = base_seed + i * 17
        canvas = gen.generate(tree_type, size=size, seed=seed, season=season)

        # Slightly vary palette seed from canvas seed for color diversity
        palette_seed = seed + 1000

        img = gen.render_to_image(canvas, palette_type, seed=palette_seed)

        if display_scale > 1:
            new_sz = size * display_scale
            img = img.resize((new_sz, new_sz), Image.NEAREST)

        buf = io.BytesIO()
        img.save(buf, format="PNG")

        caption = _CAPTIONS.get(tree_type, {}).get(season, "pixel art tree")

        cfg = TREE_PALETTE_CONFIGS.get(palette_type)
        pal_name = f"tree_{cfg[0]:.0f}_{cfg[1]:.0f}" if cfg else palette_type

        sample = {
            "image_bytes": buf.getvalue(),
            "sprite_name": f"proc_{palette_type}_{i:04d}",
            "palette_name": pal_name,
            "category": "nature",
            "size": size * display_scale if display_scale > 1 else size,
            "seed": seed,
            "caption": caption,
        }
        samples.append(sample)

        if include_flipped:
            arr = np.array(img)
            arr_flip = arr[:, ::-1, :].copy()
            buf2 = io.BytesIO()
            Image.fromarray(arr_flip, mode="RGBA").save(buf2, format="PNG")
            samples.append({
                **sample,
                "image_bytes": buf2.getvalue(),
                "sprite_name": f"proc_{palette_type}_{i:04d}_flipped",
                "seed": seed + 50000,
                "caption": caption + " flipped",
            })

    return samples


def generate_all_tree_batches(
    size: int = 16,
    n_per_type: int = 15,
    base_seed: int = 42,
) -> list[dict]:
    """Generate training samples for ALL tree palette types."""
    all_samples = []
    for palette_type in TREE_PALETTE_CONFIGS:
        batch = generate_tree_batch(
            palette_type, size=size, n=n_per_type, base_seed=base_seed
        )
        all_samples.extend(batch)
    return all_samples
