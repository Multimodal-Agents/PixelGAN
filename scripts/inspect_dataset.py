#!/usr/bin/env python3
"""Inspect parquet dataset compression stats."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import io
import os
import pandas as pd
import numpy as np
from PIL import Image

path = sys.argv[1] if len(sys.argv) > 1 else "datasets/sprites/sprites_seed_64x64_trees.parquet"
df = pd.read_parquet(path)

print(f"Columns:   {df.columns.tolist()}")
print(f"Rows:      {len(df)}")
print(f"Parquet file size: {os.path.getsize(path) / 1024:.1f} KB")

# Image byte sizes
img_sizes = [len(bytes(row['image'])) for _, row in df.iterrows()]
print(f"\nPNG bytes/image:  min={min(img_sizes):,}  max={max(img_sizes):,}  avg={int(np.mean(img_sizes)):,}")
print(f"Total image data: {sum(img_sizes)/1024:.1f} KB in parquet")

# Decode one image and inspect
img = Image.open(io.BytesIO(bytes(df.iloc[0]['image'])))
arr = np.array(img)
print(f"\nImage shape: {arr.shape}  dtype: {arr.dtype}")
print(f"Mode:        {img.mode}")

if arr.ndim == 3 and arr.shape[2] == 4:
    flat = arr.reshape(-1, 4)
    visible = flat[flat[:, 3] > 0]
    unique_colors = set(map(tuple, visible[:, :3]))
    print(f"Unique RGBA values across all pixels: {len(set(map(tuple, flat)))}")
    print(f"Unique RGB in visible pixels:         {len(unique_colors)}")
    print(f"Transparent pixel ratio:              {(flat[:,3]==0).mean()*100:.1f}%")
    print(f"\nSample visible RGB values:")
    for c in sorted(unique_colors)[:12]:
        print(f"  {c}")

# Check palette index structure across 10 images
print("\n--- Palette index check (first 10 images) ---")
all_unique = set()
for _, row in df.head(10).iterrows():
    img2 = Image.open(io.BytesIO(bytes(row['image'])))
    a = np.array(img2.convert("RGBA")).reshape(-1, 4)
    vis = a[a[:,3]>0]
    for c in vis[:,:3]:
        all_unique.add(tuple(c))
print(f"Total unique RGB colors across 10 images: {len(all_unique)}")
print(f"Colors (max 24): {sorted(all_unique)[:24]}")

# Estimate compressed sizes
sample_arr = np.array(Image.open(io.BytesIO(bytes(df.iloc[0]['image']))).convert("RGB"))
np_raw = sample_arr.nbytes
print(f"\n--- Size comparison for one 64x64 image ---")
print(f"  Raw float32 RGB (model input):   {sample_arr.astype(np.float32).nbytes:,} bytes")
print(f"  Raw uint8 RGB:                   {np_raw:,} bytes")
print(f"  PNG (current in parquet):        {len(bytes(df.iloc[0]['image'])):,} bytes")

# NPZ
buf = io.BytesIO()
np.savez_compressed(buf, img=sample_arr)
print(f"  NPZ (gzip uint8 RGB):            {buf.tell():,} bytes")

# Palette-indexed: if truly 8 colors, store uint8 index map + 8x3 palette
arr_rgba = np.array(Image.open(io.BytesIO(bytes(df.iloc[0]['image']))))
flat4 = arr_rgba.reshape(-1, 4)
palette_colors = list({tuple(c) for c in flat4[:,:3]})
print(f"  Palette colors in this image:    {len(palette_colors)}")
if len(palette_colors) <= 16:
    idx_map = np.zeros(64*64, dtype=np.uint8)
    for i, px in enumerate(flat4[:,:3]):
        idx_map[i] = palette_colors.index(tuple(px))
    alpha_map = (flat4[:,3] > 0).astype(np.uint8)
    palette_arr = np.array(palette_colors, dtype=np.uint8)
    buf2 = io.BytesIO()
    np.savez_compressed(buf2, idx=idx_map.reshape(64,64), alpha=alpha_map.reshape(64,64), palette=palette_arr)
    print(f"  NPZ palette-indexed (idx+alpha+palette): {buf2.tell():,} bytes")
