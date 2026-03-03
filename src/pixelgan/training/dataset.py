"""
PixelGAN Dataset loaders — Parquet-based.

Three dataset types, each using Parquet with exactly 2 columns:

  1. seed_to_image:
       Column "seed"   (int64)  — reproducible random seed
       Column "image"  (bytes)  — PNG/raw image bytes

  2. text_to_image:
       Column "caption" (string) — text description of the image
       Column "image"   (bytes)  — PNG/raw image bytes

  3. image_to_image:
       Column "source"  (bytes)  — source PNG image bytes
       Column "target"  (bytes)  — target PNG image bytes

The simple 2-column design makes it trivially easy to create datasets with
any image generation tool or just `pandas.DataFrame.to_parquet()`.

Example of creating a training dataset:
    import pandas as pd
    df = pd.DataFrame({"seed": seeds, "image": image_bytes_list})
    df.to_parquet("my_sprites.parquet")
"""

from __future__ import annotations

import io
import math
import random
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from PIL import Image

# Lazy import: indexed_format is only available when pyarrow is installed.
# We import at module level here so dataset.py can detect the format early;
# if pyarrow is absent the standard PNG-parquet path still works fine.
try:
    from ..data.indexed_format import (
        is_indexed_parquet,
        decode_indexed_row,
    )
    _INDEXED_FORMAT_AVAILABLE = True
except ImportError:
    _INDEXED_FORMAT_AVAILABLE = False
    def is_indexed_parquet(path) -> bool:          # type: ignore[misc]
        return False
    def decode_indexed_row(row, *a, **kw):          # type: ignore[misc]
        raise RuntimeError("pyarrow required for indexed parquet")


# ---------------------------------------------------------------------------
# Dataset type definitions
# ---------------------------------------------------------------------------

DATASET_TYPES = {
    "seed": {
        "columns": ["seed", "image"],
        "description": "Seed->Image: maps int seeds to pixel art images",
    },
    "text": {
        "columns": ["caption", "image"],
        "description": "Text->Image: maps text captions to pixel art images",
    },
    "image_pair": {
        "columns": ["source", "target"],
        "description": "Image->Image: source/target image pairs for translation",
    },
}


# ---------------------------------------------------------------------------
# Image loading utilities
# ---------------------------------------------------------------------------

def decode_image(
    image_bytes: bytes,
    target_size: int,
    channels: int = 3,
    normalize: bool = True,
    bg_rgb: tuple = (40, 40, 40),
) -> np.ndarray:
    """
    Decode PNG bytes to a numpy array.

    Args:
        image_bytes: Raw PNG bytes
        target_size: Resize to (target_size, target_size) using NEAREST neighbor
        channels: 3=RGB, 4=RGBA. When 3 and source has alpha, the sprite is
                  composited onto bg_rgb before conversion so transparent regions
                  become the background colour instead of being discarded to black.
        normalize: If True, output float32 in [-1, 1]; else uint8 in [0, 255]
        bg_rgb: Background colour (R, G, B) used when compositing RGBA -> RGB.
                Default (40, 40, 40) is a dark neutral that doesn’t bias colours.

    Returns:
        np.ndarray [target_size, target_size, channels]
    """
    img = Image.open(io.BytesIO(image_bytes))

    if channels == 4:
        img = img.convert("RGBA")
    else:
        # Composite alpha onto background before RGB conversion so transparent
        # pixels become bg_rgb rather than opaque-black (PIL’s default).
        if img.mode in ("RGBA", "LA", "PA"):
            bg = Image.new("RGBA", img.size, bg_rgb + (255,))
            img = img.convert("RGBA")
            bg.alpha_composite(img)
            img = bg.convert("RGB")
        else:
            img = img.convert("RGB")

    # Nearest-neighbor resize (critical for pixel art!)
    if img.size != (target_size, target_size):
        img = img.resize((target_size, target_size), Image.NEAREST)

    arr = np.array(img, dtype=np.uint8)  # [H, W, channels]

    if normalize:
        arr = arr.astype(np.float32) / 127.5 - 1.0  # [0,255] -> [-1, 1]

    return arr


def encode_image(
    array: np.ndarray,
    denormalize: bool = True,
) -> bytes:
    """
    Encode numpy array to PNG bytes.

    Args:
        array: [H, W, C] float32 in [-1,1] or uint8 in [0,255]
        denormalize: If True, treat input as float32 [-1,1]

    Returns:
        PNG bytes
    """
    if denormalize:
        arr = ((array + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    else:
        arr = array.astype(np.uint8)

    channels = arr.shape[2] if arr.ndim == 3 else 1
    mode = "RGBA" if channels == 4 else ("RGB" if channels == 3 else "L")
    img = Image.fromarray(arr, mode=mode)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def tokenize_text(
    text: str,
    max_length: int = 64,
    vocab_size: int = 1024,
) -> np.ndarray:
    """
    Simple character-level tokenizer for text conditioning.

    Uses a fixed vocabulary: printable ASCII + special tokens.
    This is intentionally simple — for production you'd want SentencePiece.

    Args:
        text: Input text string
        max_length: Sequence length (truncate or pad)
        vocab_size: Not used currently (character vocab is ~128)

    Returns:
        np.ndarray [max_length] int32 token indices
    """
    PAD_TOKEN = 0
    BOS_TOKEN = 1  # beginning of sequence
    EOS_TOKEN = 2  # end of sequence
    CHAR_OFFSET = 3  # printable chars start at 3

    # Lowercase and clean
    text = text.lower().strip()[:max_length - 2]  # leave room for BOS/EOS

    tokens = [BOS_TOKEN]
    for c in text:
        if ord(c) < 128:  # ASCII only
            tokens.append(min(ord(c) + CHAR_OFFSET, vocab_size - 1))
        else:
            tokens.append(CHAR_OFFSET)  # unknown char
    tokens.append(EOS_TOKEN)

    # Pad or truncate to max_length
    tokens = tokens[:max_length]
    tokens += [PAD_TOKEN] * (max_length - len(tokens))

    return np.array(tokens, dtype=np.int32)


# ---------------------------------------------------------------------------
# Parquet dataset classes
# ---------------------------------------------------------------------------

class ParquetDataset:
    """
    Base class for parquet-backed datasets.

    Lazily loads rows and converts to model-ready numpy arrays.
    Supports multiple parquet files (shards) in a directory.
    """

    DATASET_TYPE = "base"
    COLUMNS: list[str] = []

    def __init__(
        self,
        path: str | Path,
        image_size: int = 32,
        image_channels: int = 4,
        split: str = "train",
        train_ratio: float = 0.9,
        seed: int = 42,
    ):
        """
        Args:
            path: Path to .parquet file or directory of .parquet files
            image_size: Target image resolution
            image_channels: 3=RGB, 4=RGBA
            split: 'train' or 'val'
            train_ratio: Fraction of data for training
            seed: Shuffle seed
        """
        self.image_size = image_size
        self.image_channels = image_channels
        self.split = split
        self.train_ratio = train_ratio
        self._seed = seed

        # Load parquet files
        path = Path(path)
        if path.is_file():
            parquet_files = [path]
        elif path.is_dir():
            parquet_files = sorted(path.glob("*.parquet"))
        else:
            raise FileNotFoundError(f"No parquet files found at: {path}")

        if not parquet_files:
            raise FileNotFoundError(f"No .parquet files found at: {path}")

        # Load all data into memory (for small pixel art datasets this is fine)
        # For large datasets, use lazy loading
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required: pip install pandas pyarrow")

        dfs = []
        for f in parquet_files:
            df = pd.read_parquet(f, columns=self.COLUMNS)
            dfs.append(df)

        self._df = pd.concat(dfs, ignore_index=True)

        # Validate columns
        for col in self.COLUMNS:
            if col not in self._df.columns:
                raise ValueError(
                    f"Expected column '{col}' in parquet file. "
                    f"Got: {list(self._df.columns)}. "
                    f"Dataset type '{self.DATASET_TYPE}' requires: {self.COLUMNS}"
                )

        # Train/val split
        rng = random.Random(seed)
        indices = list(range(len(self._df)))
        rng.shuffle(indices)
        n_train = int(len(indices) * train_ratio)
        if split == "train":
            self._indices = indices[:n_train]
        else:
            self._indices = indices[n_train:]

    def __len__(self) -> int:
        return len(self._indices)

    def shuffle(self, seed: Optional[int] = None) -> None:
        rng = random.Random(seed or self._seed)
        rng.shuffle(self._indices)


class SeedDataset(ParquetDataset):
    """
    Seed->Image dataset.

    Supports **two** parquet schemas transparently:

    1. Standard (PNG-image) schema:
          seed  (int64)   — reproducible random seed
          image (bytes)   — PNG image bytes

    2. Indexed (palette-compressed) schema produced by convert_to_indexed.py:
          seed       (int64)  — reproducible random seed
          index_map  (bytes)  — uint8 [H,W] palette index map (raw bytes)
          palette_data (bytes) — uint8 [N,3] RGB palette (raw bytes)
          n_colors   (int64)  — number of palette entries used

    When the indexed schema is detected the dataset automatically uses
    `decode_indexed_row()` and returns an extra ``palette`` key so the
    trainer can feed it to `PaletteLookup` (Option A) without any changes
    to the training script itself.
    """
    DATASET_TYPE = "seed"
    # Accept both schemas — pandas will load whichever columns exist.
    COLUMNS = ["seed", "image"]  # overridden below when indexed

    def __init__(self, *args, n_palette_slots: int = 16, **kwargs):
        """
        Extra arg:
            n_palette_slots: Palette array size (padded/truncated to this).
                             Must match the generator's n_palette_colors.
                             Only used in indexed mode.
        """
        self.n_palette_slots = n_palette_slots

        # Detect schema *before* calling super().__init__ so we can set
        # COLUMNS correctly and avoid a missing-column error.
        from pathlib import Path as _Path
        _path = _Path(args[0]) if args else _Path(kwargs["path"])
        self._is_indexed = (
            _INDEXED_FORMAT_AVAILABLE and is_indexed_parquet(_path)
        )
        if self._is_indexed:
            self.__class__.COLUMNS = [
                "seed", "index_map", "palette_data", "n_colors"
            ]
        else:
            self.__class__.COLUMNS = ["seed", "image"]

        super().__init__(*args, **kwargs)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns a dict with:
          Always:
            ``z_seed``  int
            ``image``   [H, W, C] float32 in [-1, 1]
          Only in indexed mode:
            ``palette`` [n_palette_slots, 3] float32 in [-1, 1]
        """
        row_idx = self._indices[idx]
        row = self._df.iloc[row_idx]

        if self._is_indexed:
            image, palette = decode_indexed_row(
                row,
                target_size=self.image_size,
                n_palette_slots=self.n_palette_slots,
            )
            return {
                "z_seed":  int(row["seed"]),
                "image":   image.astype(np.float32),
                "palette": palette.astype(np.float32),
            }
        else:
            image = decode_image(
                bytes(row["image"]), self.image_size, self.image_channels
            )
            return {
                "z_seed": int(row["seed"]),
                "image":  image.astype(np.float32),
            }

    def get_batch(self, batch_size: int, start_idx: int = 0) -> dict:
        """Get a batch of samples as numpy arrays."""
        images, seeds = [], []
        palettes = [] if self._is_indexed else None

        for i in range(batch_size):
            idx = (start_idx + i) % len(self)
            sample = self[idx]
            images.append(sample["image"])
            seeds.append(sample["z_seed"])
            if palettes is not None:
                palettes.append(sample["palette"])

        batch = {
            "image":  np.stack(images),                  # [B, H, W, C]
            "z_seed": np.array(seeds, dtype=np.int64),   # [B]
        }
        if palettes is not None:
            batch["palette"] = np.stack(palettes)        # [B, N, 3]
        return batch


class TextDataset(ParquetDataset):
    """
    Text->Image dataset.

    Parquet format:
      - caption (string): Text description of the pixel art image
      - image   (bytes):  PNG image bytes

    Used for text-guided generation: model learns to map text descriptions
    to pixel art images.

    Example captions:
      "galaga bee alien purple with gold eyes"
      "zelda link character top-down green tunic"
      "pacman ghost red blinky"
    """
    DATASET_TYPE = "text"
    COLUMNS = ["caption", "image"]

    def __init__(self, *args, text_max_length: int = 64,
                 text_vocab_size: int = 1024, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_max_length = text_max_length
        self.text_vocab_size = text_vocab_size

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            dict with:
              - 'tokens': [seq_len] int32 text tokens
              - 'caption': str raw text
              - 'image': [H, W, C] float32 in [-1, 1]
        """
        row_idx = self._indices[idx]
        row = self._df.iloc[row_idx]

        caption = str(row["caption"])
        tokens = tokenize_text(caption, self.text_max_length, self.text_vocab_size)
        image = decode_image(bytes(row["image"]), self.image_size, self.image_channels)

        return {
            "tokens": tokens.astype(np.int32),
            "caption": caption,
            "image": image.astype(np.float32),
        }

    def get_batch(self, batch_size: int, start_idx: int = 0) -> dict:
        images, tokens, captions = [], [], []
        for i in range(batch_size):
            sample = self[(start_idx + i) % len(self)]
            images.append(sample["image"])
            tokens.append(sample["tokens"])
            captions.append(sample["caption"])
        return {
            "image": np.stack(images),
            "tokens": np.stack(tokens),
            "caption": captions,
        }


class ImagePairDataset(ParquetDataset):
    """
    Image->Image dataset.

    Parquet format:
      - source (bytes): Source PNG image bytes
      - target (bytes): Target PNG image bytes

    Used for image-to-image translation: model learns to transform
    source style images into target style images.

    Examples:
      - Sketch -> colored pixel art
      - Daytime scene -> night scene
      - Low-detail -> high-detail
      - Style A -> Style B (CycleGAN-style)
    """
    DATASET_TYPE = "image_pair"
    COLUMNS = ["source", "target"]

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            dict with:
              - 'source': [H, W, C] float32 source image
              - 'target': [H, W, C] float32 target image
        """
        row_idx = self._indices[idx]
        row = self._df.iloc[row_idx]

        source = decode_image(bytes(row["source"]), self.image_size, self.image_channels)
        target = decode_image(bytes(row["target"]), self.image_size, self.image_channels)

        return {
            "source": source.astype(np.float32),
            "target": target.astype(np.float32),
        }

    def get_batch(self, batch_size: int, start_idx: int = 0) -> dict:
        sources, targets = [], []
        for i in range(batch_size):
            sample = self[(start_idx + i) % len(self)]
            sources.append(sample["source"])
            targets.append(sample["target"])
        return {
            "source": np.stack(sources),
            "target": np.stack(targets),
        }


# ---------------------------------------------------------------------------
# Dataset factory
# ---------------------------------------------------------------------------

def load_dataset(
    path: str | Path,
    dataset_type: str,
    image_size: int = 32,
    image_channels: int = 4,
    split: str = "train",
    **kwargs,
) -> ParquetDataset:
    """
    Load a parquet dataset by type.

    Args:
        path: Path to .parquet file or directory
        dataset_type: 'seed', 'text', or 'image_pair'
        image_size: Target resolution
        image_channels: 3 or 4
        split: 'train' or 'val'
        **kwargs: Additional dataset-specific kwargs

    Returns:
        ParquetDataset subclass

    Example:
        ds = load_dataset("sprites.parquet", "seed", image_size=32)
        batch = ds.get_batch(16)
    """
    cls_map = {
        "seed": SeedDataset,
        "text": TextDataset,
        "image_pair": ImagePairDataset,
    }

    if dataset_type not in cls_map:
        raise ValueError(
            f"Unknown dataset_type: {dataset_type!r}. "
            f"Use one of: {list(cls_map.keys())}"
        )

    return cls_map[dataset_type](
        path=path,
        image_size=image_size,
        image_channels=image_channels,
        split=split,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Infinite data iterator
# ---------------------------------------------------------------------------

def infinite_loader(
    dataset: ParquetDataset,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 42,
) -> Iterator[dict]:
    """
    Infinite iterator that loops through the dataset repeatedly.
    Re-shuffles on each epoch.

    Args:
        dataset: A ParquetDataset
        batch_size: Batch size
        shuffle: Whether to shuffle each epoch
        seed: Shuffle seed (increments each epoch)

    Yields:
        Batch dicts
    """
    epoch = 0
    while True:
        if shuffle:
            dataset.shuffle(seed + epoch)

        n_batches = math.ceil(len(dataset) / batch_size)
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            yield dataset.get_batch(batch_size, start)

        epoch += 1


# ---------------------------------------------------------------------------
# Dataset creation utilities
# ---------------------------------------------------------------------------

def create_seed_dataset(
    image_bytes_list: list[bytes],
    seeds: Optional[list[int]] = None,
    output_path: str = "dataset.parquet",
) -> Path:
    """
    Create a seed->image parquet dataset from image bytes.

    Args:
        image_bytes_list: List of PNG image bytes
        seeds: List of integer seeds (auto-generated if None)
        output_path: Where to save the parquet file

    Returns:
        Path to created parquet file
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas required: pip install pandas pyarrow")

    if seeds is None:
        seeds = list(range(len(image_bytes_list)))

    df = pd.DataFrame({
        "seed": seeds,
        "image": [bytes(b) for b in image_bytes_list],
    })

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)

    print(f"Saved seed dataset: {len(df)} samples -> {output}")
    return output


def create_text_dataset(
    image_bytes_list: list[bytes],
    captions: list[str],
    output_path: str = "dataset.parquet",
) -> Path:
    """Create a text->image parquet dataset."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas required: pip install pandas pyarrow")

    assert len(image_bytes_list) == len(captions)

    df = pd.DataFrame({
        "caption": captions,
        "image": [bytes(b) for b in image_bytes_list],
    })

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)

    print(f"Saved text dataset: {len(df)} samples -> {output}")
    return output


def create_image_pair_dataset(
    source_bytes: list[bytes],
    target_bytes: list[bytes],
    output_path: str = "dataset.parquet",
) -> Path:
    """Create a source->target image pair parquet dataset."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas required: pip install pandas pyarrow")

    assert len(source_bytes) == len(target_bytes)

    df = pd.DataFrame({
        "source": [bytes(b) for b in source_bytes],
        "target": [bytes(b) for b in target_bytes],
    })

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)

    print(f"Saved image-pair dataset: {len(df)} samples -> {output}")
    return output
