# Contributing to PixelGAN

## Branch strategy

```
main        ← stable releases only, tagged (e.g. v0.1.0)
  └── nightly  ← integration branch, continuously updated
        ├── feature/<short-description>
        ├── fix/<short-description>
        └── chore/<short-description>
```

### Rules

| Branch | Purpose | Merges from | Merges into |
|---|---|---|---|
| `main` | Stable, releasable code | `nightly` only | — |
| `nightly` | Integration / testing | `feature/*` `fix/*` `chore/*` | `main` |
| `feature/*` | New capabilities | `nightly` (rebase before PR) | `nightly` |
| `fix/*` | Bug fixes | `nightly` (rebase before PR) | `nightly` |
| `chore/*` | Deps, CI, docs, refactors | `nightly` (rebase before PR) | `nightly` |

**Never commit directly to `main` or `nightly`.**

---

## Day-to-day workflow

### 1 — Start a new piece of work

```bash
git checkout nightly
git pull origin nightly

# Pick the right prefix:
git checkout -b feature/my-new-thing    # new capability
git checkout -b fix/broken-sprite-alpha # bug fix
git checkout -b chore/update-jax        # maintenance
```

### 2 — Work, commit often

```bash
git add -p          # stage hunks, never "git add ."
git commit -m "fix: correct XOR-shift wrapping for seed=0"
```

Commit message format: `<type>: <short imperative description>`

| Type | When |
|---|---|
| `feat` | New feature |
| `fix` | Bug fix |
| `chore` | Deps, CI, tooling, docs |
| `refactor` | Code restructure (no behaviour change) |
| `test` | Tests only |

### 3 — Keep your branch up to date with nightly

```bash
git fetch origin
git rebase origin/nightly
```

Resolve any conflicts, then `git rebase --continue`.

### 4 — Push and open a PR into nightly

```bash
git push -u origin feature/my-new-thing
# Open PR on GitHub: base = nightly
```

### 5 — Promoting nightly → main (release)

When `nightly` is stable and tested:

```bash
git checkout main
git pull origin main
git merge --no-ff nightly -m "release: vX.Y.Z"
git tag vX.Y.Z
git push origin main --tags
```

---

## Dataset & checkpoint hygiene

Large files are excluded by `.gitignore` and must **never** be committed:

- `datasets/sprites/*.parquet` — regenerate with `scripts/generate_zzsprites.py`
- `runs/` / `outputs/` — training artifacts
- `*.pkl` / `*.pt` / `*.ckpt` / `*.safetensors` — model weights

If you need to share a checkpoint, attach it to a GitHub Release, not a commit.
