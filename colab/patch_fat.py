"""Patch EleutherAI/features-across-time so `train_vision.py` runs on a single Colab GPU.

Usage:  python colab/patch_fat.py /path/to/features-across-time

Only `scripts/inference/train_vision.py` is changed. Every change is controlled by an
environment variable so the notebook can configure runs without editing code again:

  FAT_OUTPUT_DIR   where checkpoints go           (default: runs/<dataset>/<net>)
  FAT_MAX_STEPS    training steps                 (default: 65536, as in the original)
  FAT_WARMUP       LR warmup steps                (default: 2000, as in the original)
  FAT_BATCH        per-device batch size          (default: 128, as in the original)
  FAT_EVAL_SETS    comma list of eval sets        (default: real,gaussian,independent,cqn)

What the patch fixes:
  1. `.cuda()` calls -> auto device (CUDA if present, else CPU) so it also runs locally.
  2. bf16 / tf32 only when the GPU supports them (Colab's T4 does not -> fp32).
  3. Reporting to wandb disabled.
  4. The `maxent` / `shift` eval sets need files built by another script under a folder
     name that does not match (`shifted-data/` vs `data/`), so they are skipped unless
     requested. `got` needs ~4 GB of optimal-transport maps, so it is opt-in too.
  5. Output dir includes whatever FAT_OUTPUT_DIR says, so real- and fake-trained runs
     no longer overwrite each other.
"""

import sys
from pathlib import Path


def replace_once(src: str, old: str, new: str) -> str:
    count = src.count(old)
    if count != 1:
        raise SystemExit(f"Patch target found {count} times (expected 1):\n{old}")
    return src.replace(old, new)


def replace_between(src: str, start: str, end: str, new: str) -> str:
    i = src.find(start)
    j = src.find(end, i)
    if i < 0 or j < 0:
        raise SystemExit(f"Could not find block starting with:\n{start}")
    return src[:i] + new + src[j + len(end):]


def main(repo: Path) -> None:
    path = repo / "scripts" / "inference" / "train_vision.py"
    src = path.read_text()
    if "# PATCHED-FOR-COLAB" in src:
        print("Already patched:", path)
        return

    # 1. Device helper
    src = replace_once(
        src,
        "from transformers.modeling_outputs import ModelOutput\n",
        "from transformers.modeling_outputs import ModelOutput\n\n"
        "# PATCHED-FOR-COLAB\n"
        "DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
        "_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()\n"
        "_TF32 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8\n\n\n"
        "def _images_labels(d, img_col, label_col):\n"
        "    # datasets>=4 returns a lazy Column for d[col] and channel-first images;\n"
        "    # older versions return a channel-last tensor. Handle both.\n"
        "    batch = d[:]\n"
        "    X = assert_type(Tensor, batch[img_col]).div(255)\n"
        "    if X.shape[-1] in (1, 3):\n"
        "        X = rearrange(X, 'n h w c -> n c h w')\n"
        "    return X, assert_type(Tensor, batch[label_col])\n",
    )
    src = replace_once(
        src,
        "    X = assert_type(Tensor, train[img_col]).div(255)\n"
        '    X = rearrange(X, "n h w c -> n c h w")\n'
        "    Y = assert_type(Tensor, train[label_col])\n",
        "    X, Y = _images_labels(train, img_col, label_col)\n",
    )
    src = replace_once(
        src,
        "fitter = QuadraticFitter.fit(X.flatten(1).cuda(), Y.cuda())",
        "fitter = QuadraticFitter.fit(X.flatten(1).to(DEVICE), Y.to(DEVICE))",
    )

    # 4. Build only the requested eval sets
    src = replace_between(
        src,
        '    cache = Path.cwd() / "editor-cache"',
        '        "shift": shifted,\n    }\n',
        '''    with val.formatted_as("torch"):
        X, Y = _images_labels(val, img_col, label_col)

    wanted = os.environ.get("FAT_EVAL_SETS", "real,gaussian,independent,cqn").split(",")
    val_sets = {}
    for name in wanted:
        if name == "real":
            val_sets[name] = val
        elif name == "gaussian":
            val_sets[name] = gaussian
        elif name == "independent":
            val_sets[name] = IndependentCoordinateSampler(class_probs, normalizer, len(val))
        elif name == "cqn":
            val_sets[name] = QuantileNormalizedDataset(class_probs, normalizer, X, Y)
        elif name == "got":
            cache = Path.cwd() / "editor-cache" / f"{dataset_str.replace('/', '--')}.pkl"
            if cache.exists():
                with open(cache, "rb") as f:
                    editor = pickle.load(f)
            else:
                print("Computing optimal transport maps...")
                editor = fitter.editor("cpu")
                cache.parent.mkdir(exist_ok=True)
                with open(cache, "wb") as f:
                    pickle.dump(editor, f)
            val_sets[name] = ConceptEditedDataset(class_probs, editor, X, Y)
        elif name in ("maxent", "shift"):
            prefix = "dury" if name == "maxent" else "shifted"
            d = load_from_disk(Path.cwd() / f"shifted-data/{prefix}-{dataset_str.replace('/', '--')}.hf")
            d.set_format("torch", columns=["pixel_values", "label"])
            val_sets[name] = d
        else:
            raise ValueError(f"Unknown eval set {name}")
    print("Eval sets:", list(val_sets))
''',
    )

    # 2, 3, 5. Training arguments
    src = replace_once(
        src,
        'output_dir=f"runs/{ds_str}/{net_str}",',
        'output_dir=os.environ.get("FAT_OUTPUT_DIR", f"runs/{ds_str}/{net_str}"),',
    )
    src = replace_once(src, "bf16=True,", "bf16=_BF16,")
    src = replace_once(src, "tf32=True,", "tf32=_TF32,")
    src = replace_once(
        src,
        "dataloader_num_workers=8,",
        "dataloader_num_workers=min(8, os.cpu_count() or 1),",
    )
    src = replace_once(
        src,
        "max_steps=2**16,",
        'max_steps=int(os.environ.get("FAT_MAX_STEPS", 2**16)),',
    )
    src = replace_once(
        src,
        "per_device_train_batch_size=128,",
        'per_device_train_batch_size=int(os.environ.get("FAT_BATCH", 128)),\n'
        '        report_to="none",',
    )
    src = replace_once(
        src,
        "warmup_steps=2000,",
        'warmup_steps=int(os.environ.get("FAT_WARMUP", 2000)),',
    )

    path.write_text(src)
    print("Patched:", path)


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "features-across-time"))
