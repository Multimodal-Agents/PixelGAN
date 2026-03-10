# PixelGAN Training Cheat Sheet

## Paths & Env
| Thing | Value |
|-------|-------|
| Root | `m:\_tools\docs\stable-matrix-gan-library\` |
| Root (WSL path) | `/mnt/m/_tools/docs/stable-matrix-gan-library/` |
| Venv (WSL) | `.venv-wsl/bin/python` |
| Venv (Win) | `.venv\Scripts\python` |
| Dataset dir | `datasets/sprites/` |

### Navigate there from WSL

```bash
cd /mnt/m/_tools/docs/stable-matrix-gan-library
```

> Windows drives in WSL are at `/mnt/<driveletter>` (lowercase). `M:` → `/mnt/m/`.

---

## Step 1 — Generate 64×64 Trees

```bash
.venv-wsl/bin/python scripts/generate_trees.py \
  --size 64 --n-per-type 40
```

Writes `datasets/sprites/sprites_seed_64x64_trees.parquet` (~2340 samples × 40/type × flips).

---

## Step 2 — Convert to Indexed (Global Palette, k=4)

Use a **fixed k=4** so it matches `--palette-colors 5` exactly.

```bash
.venv-wsl/bin/python scripts/convert_to_indexed.py \
  --input datasets/sprites/sprites_seed_64x64_trees.parquet \
  --global-palette --k 4
```

Result: `sprites_seed_64x64_trees_indexed_global.parquet`

> **Why k=4?** `--palette-colors 5` = 4 visible slots + 1 transparent (slot 0).
> If you use `--auto-k` it may pick k≠4 and training will misalign.

---

## Step 3 — Train (64×64 Trees)

```bash
rm -rf runs/pixelgan

.venv-wsl/bin/python scripts/train.py \
  --size 64 \
  --dataset datasets/sprites/sprites_seed_64x64_trees_indexed_global.parquet \
  --output-mode palette_indexed \
  --palette-colors 5 \
  --batch-size 8 \
  --log-every 100 \
  --sample-every 500 \
  --no-prealloc
```

> `--batch-size 8`  — better stability than default 4; use `--no-prealloc` for memory.
> Logs print "*** COLLAPSE? ***" if G+D both drop below 0.05 — restart immediately if so.
> Samples saved to `runs/pixelgan/samples_NNNNNN.png`

---

## Step 4 — Train on ZzSprites (additional data source)

ZzSprite is a second procedural generator (ported from ZzSprite.js by Frank Force).
Generates organic blob-like sprites — complements the tree dataset.

```bash
# Generate ZzSprite dataset
.venv-wsl/bin/python scripts/generate_zzsprites.py \
  --size 64 --n 600 --modes 0 1

# Train on it (RGB mode, no palette needed)
rm -rf runs/pixelgan
.venv-wsl/bin/python scripts/train.py \
  --size 64 \
  --channels 4 \
  --dataset datasets/sprites/sprites_zzsprite_64x64.parquet \
  --output-mode rgb \
  --batch-size 8 \
  --no-prealloc
```

---

## Known Issues & Fixes Applied

| Bug | Symptom | Fix |
|-----|---------|-----|
| `TracerBoolConversionError` in `losses.py` | Crash during JIT of first palette_indexed step with "Attempted boolean conversion of traced array" | Removed `and lambda_entropy > 0` guard — multiply unconditionally by `lambda_entropy` (0·x=0 is a JAX no-op). |
| Speed metric includes JIT compilation | `kimg/s` looks 5–20× too slow on short runs | Normal: JIT takes 166s (8×8) to 422s (64×64) on first run, then amortizes. Restart shows true speed. |

---

## Kill & Restart Training

```bash
pkill -9 -f 'scripts/train.py'
rm -rf runs/pixelgan
# then rerun Step 3
```

---

## What Healthy Training Looks Like

| Metric | Healthy | Collapsed |
|--------|---------|-----------|
| G loss | 0.5 – 2.0 | < 0.05 (prints *** COLLAPSE ***) |
| D loss | 0.5 – 2.0 | < 0.05 (prints *** COLLAPSE ***) |
| ADA p | gradually rising | stuck near 0 or at 1.0 |
| lambda_entropy | 0.05 → 0 by step 15k | stuck at 0.05 (old bug, fixed) |
| entropy key in metrics | present in palette mode | absent in rgb mode |
| Samples step ~500 | colored blobs / shapes | blank / solid dark gray |

---

## Key Flags Reference

| Flag | Default | Notes |
|------|---------|-------|
| `--size` | — | Image size (8/16/32/64/128) |
| `--channels` | 3 | 3=RGB, 4=RGBA |
| `--output-mode` | `rgb` | Use `palette_indexed` for palette mode |
| `--palette-colors` | 8 | k+1 total slots (k visible + 1 transparent); use 5 for k=4 |
| `--batch-size` | size-dependent | 8 recommended for 64×64; 4 is the minimum |
| `--log-every` | 100 | Steps between metric prints |
| `--sample-every` | 500 | Steps between PNG saves |
| `--no-prealloc` | off | Disables JAX GPU memory preallocation (recommended) |
| `--gpu-mem-fraction` | 0.75 | Fraction of VRAM JAX may use |

---

## Data Generation Reference

### Trees (primary dataset)

```bash
# All sizes at once
.venv-wsl/bin/python scripts/generate_trees.py --all-sizes --n-per-type 40

# Preview only (no parquet)
.venv-wsl/bin/python scripts/generate_trees.py --size 64 --preview-only

# List all tree types
.venv-wsl/bin/python scripts/generate_trees.py list
```

### ZzSprite (secondary dataset)

```bash
# All sizes
.venv-wsl/bin/python scripts/generate_zzsprites.py --all-sizes --n 400

# Preview only
.venv-wsl/bin/python scripts/generate_zzsprites.py --size 64 --preview-only

# Colored sprites only (mode 0)
.venv-wsl/bin/python scripts/generate_zzsprites.py --size 64 --modes 0 --n 600

# Text-captioned (for text-conditioned training)
.venv-wsl/bin/python scripts/generate_zzsprites.py --size 32 --dataset-mode text
```

---

## Architecture Overview

```
Generator  4×4 const → SynthesisBlocks (pixel-shuffle upsample) → ToPaletteLogits
           ~400k params at 64×64 | palette-indexed or RGB output

Discriminator  FromRGB → DiscriminatorResBlocks (stride-2 downsample) → FC
               ~150k params at 64×64 | receives RGB from palette_lookup

Training   G: non-saturating loss | D: non-saturating + R1 gradient penalty
           ADA: adaptive augmentation | EMA: exponential moving average of G
```

---

## Git Branch

```
feature/indexed-palette-vqvae
```

Latest stable commit: `97fe043`
Message: `fix: decay lambda_entropy to 0 by step 15k — prevents late-training blank output collapse`

---

## Quick Debug Checklist

- **Blank samples early (step < 5k)** → check G/D loss; if both < 0.05 (log says `*** COLLAPSE ***`): kill, `rm -rf runs/`, restart with `--batch-size 8`.
- **Blank samples late (step > 15k)** → lambda_entropy decay bug (old code); fixed in `97fe043`.
- **`ArrowInvalid: No match for palette_data`** → wrong dataset (pre-conversion). Run Step 2 first.
- **`FileNotFoundError`** → dataset path wrong or Step 2 not run.
- **`ScopeParamNotFoundError`** → Flax param registration bug; check git is on latest commit.
- **OOM / CUDA crash** → add `--no-prealloc` or reduce `--batch-size`.
- **All sprites same dark gray** → generator stuck outputting slot 0 (transparent). Check entropy decay is working (lambda_entropy should print decreasing values toward 0).
- **Wrong colors** → palette mismatch: k in Step 2 must equal `--palette-colors - 1` in Step 3. Use `--k 4` in convert + `--palette-colors 5` in train.
