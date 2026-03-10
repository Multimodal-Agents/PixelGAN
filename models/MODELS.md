# PixelGAN Model Catalog

Trained checkpoints are tracked here. Checkpoints themselves are excluded from
the repo (see `.gitignore`) — attach them to GitHub Releases.

Each model follows the naming convention: `<dataset>-<version>` (e.g. `space-monsters-1`).

---

## Models

### space-monsters-1 *(in training)*

| Field | Value |
|---|---|
| Dataset | ZzSprite 32×32 — all 4 modes (colored · grayscale · silhouette · black) |
| Generator params | 2,530,121 |
| Discriminator params | 9,056,705 |
| Image size | 32×32 |
| Channels | 3 (RGB) |
| Output mode | `rgb` |
| Batch size | 32 |
| G LR / D LR | 0.0002 / 0.0002 |
| R1 γ | 10.0 |
| ADA target | 0.6 |
| Status | Smoke test ✅ → full run pending |

**Smoke test (500 steps / 16 kimg):**
- G loss: 4.2965
- D loss: 0.0984
- Speed: 0.037 kimg/s (after JIT warmup)
- Wall time: ~2m 33s train + ~4m 28s JIT compile (first run only)

**Full training target:** 10 000 steps / 320 kimg (~90 min on current GPU)

**Reproduce:**
```bash
# 1 — Generate dataset
python scripts/generate_zzsprites.py --size 32 --n 600 --modes 0 1 2 3

# 2 — Train
python scripts/train.py \
    --size 32 \
    --dataset datasets/sprites/sprites_zzsprite_32x32.parquet \
    --output runs/space-monsters-1 \
    --steps 10000 \
    --log-every 200 \
    --sample-every 500 \
    --checkpoint-every 2000 \
    --no-prealloc
```

---

## Adding a new model

1. Train using `scripts/train.py`
2. Add a row in this file with full hyperparams and final metrics
3. Export the checkpoint to `models/<name>/checkpoint/`
4. Upload the checkpoint as a GitHub Release asset
5. Link the release here
