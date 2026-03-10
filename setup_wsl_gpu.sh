#!/usr/bin/env bash
set -e
REPO=/mnt/m/_tools/docs/stable-matrix-gan-library
VENV=$REPO/.venv-wsl
PIP=$VENV/bin/pip
PY=$VENV/bin/python3

cd "$REPO"

echo "==> Creating .venv-wsl..."
python3 -m venv "$VENV"

echo "==> Upgrading pip..."
"$PIP" install --upgrade pip

echo "==> Installing JAX (CUDA 12)..."
"$PIP" install "jax[cuda12]" numpy pyarrow pandas Pillow tqdm sentencepiece matplotlib

echo "==> Installing flax + orbax without uvloop..."
"$PIP" install flax --no-deps
"$PIP" install orbax-checkpoint --no-deps
"$PIP" install msgpack rich PyYAML treescope tensorstore aiofiles humanize simplejson psutil chex toolz dataclasses-json etils jaxtyping protobuf orbax-export importlib_resources --no-deps
"$PIP" install "marshmallow>=3.18.0,<4.0.0" typing-inspect mypy-extensions optax

echo ""
echo "==> Verifying JAX sees GPU..."
"$PY" -c "import jax; print('Devices:', jax.devices())"

echo ""
echo "==> Done! To train, open WSL and run:"
echo "    source $VENV/bin/activate"
echo "    cd $REPO"
echo "    python scripts/train.py --size 8 --dataset datasets/sprites/sprites_seed_8x8.parquet --steps 500 --log-every 50 --checkpoint-every 500"
