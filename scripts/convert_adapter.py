#!/usr/bin/env python3
"""Convert Stage 2 adapter for compatibility with dual-adapter loading.

Strips the `_orig_mod.` prefix from adapter weight keys that gets added
by torch.compile() during training. Without this conversion, the adapter
cannot be loaded alongside other adapters on AutoModelForImageTextToText.

Usage:
    python scripts/convert_adapter.py [--input PATH] [--output PATH]

Example:
    python scripts/convert_adapter.py \
        --input outputs/medgemma-stage2-probabilistic/ \
        --output outputs/medgemma-stage2-converted/
"""

import argparse
import json
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file


PREFIX = "_orig_mod."


def convert_adapter(input_dir: Path, output_dir: Path) -> None:
    """Strip _orig_mod. prefix from adapter weights and save."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load adapter weights
    weights_path = input_dir / "adapter_model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"No adapter_model.safetensors in {input_dir}")

    print(f"Loading adapter from {weights_path}")
    tensors = load_file(str(weights_path))

    # Strip prefix from keys
    converted = {}
    n_renamed = 0
    for key, tensor in tensors.items():
        if key.startswith(PREFIX):
            new_key = key[len(PREFIX):]
            converted[new_key] = tensor
            n_renamed += 1
        else:
            converted[key] = tensor

    print(f"Renamed {n_renamed}/{len(tensors)} keys (stripped '{PREFIX}' prefix)")

    if n_renamed == 0:
        print("No keys needed conversion — adapter is already compatible.")
        return

    # Save converted weights
    out_weights = output_dir / "adapter_model.safetensors"
    save_file(converted, str(out_weights))
    print(f"Saved converted weights to {out_weights}")

    # Copy and update adapter config
    config_path = input_dir / "adapter_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

        # Ensure inference mode is set
        config["inference_mode"] = True

        out_config = output_dir / "adapter_config.json"
        with open(out_config, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Saved adapter config to {out_config}")

    # Copy README if present
    readme_path = input_dir / "README.md"
    if readme_path.exists():
        shutil.copy2(readme_path, output_dir / "README.md")

    # Print sample key comparison
    sample_old = list(tensors.keys())[0]
    sample_new = list(converted.keys())[0]
    print(f"\nSample key conversion:")
    print(f"  Before: {sample_old}")
    print(f"  After:  {sample_new}")

    # Verify key format matches community adapter pattern
    expected_prefix = "base_model.model.model.language_model.layers"
    if any(k.startswith(expected_prefix) for k in converted):
        print(f"\n✓ Keys match expected pattern ({expected_prefix}...)")
        print("  Compatible with dual-adapter loading on AutoModelForImageTextToText")
    else:
        print(f"\n⚠ Keys don't start with expected prefix: {expected_prefix}")
        print("  Manual verification recommended before upload")


def find_best_checkpoint(base_dir: Path) -> Path:
    """Find the best or latest checkpoint in the output directory."""
    # Check for final model (no checkpoint suffix)
    if (base_dir / "adapter_model.safetensors").exists():
        return base_dir

    # Find checkpoints
    checkpoints = sorted(base_dir.glob("checkpoint-*"),
                         key=lambda p: int(p.name.split("-")[1]))
    if checkpoints:
        return checkpoints[-1]

    raise FileNotFoundError(f"No adapter found in {base_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert Stage 2 adapter for dual-adapter loading")
    parser.add_argument("--input", type=str,
                        default="outputs/medgemma-stage2-probabilistic/adapter",
                        help="Input adapter directory (or parent with checkpoints)")
    parser.add_argument("--output", type=str,
                        default="outputs/medgemma-stage2-converted",
                        help="Output directory for converted adapter")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    # Find best checkpoint if needed
    input_dir = find_best_checkpoint(input_dir)
    print(f"Using adapter from: {input_dir}")

    convert_adapter(input_dir, output_dir)
    print(f"\nDone! Upload converted adapter with:")
    print(f"  huggingface-cli upload drdavidl/medgemma-stage2-report {output_dir}")


if __name__ == "__main__":
    main()
