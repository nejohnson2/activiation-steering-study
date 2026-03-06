"""Inspect generated outputs for qualitative review.

Usage:
    # Show all outputs for a model/persona
    python scripts/inspect_outputs.py --model llama3.1-8b --persona formal_academic

    # Show only baselines
    python scripts/inspect_outputs.py --model llama3.1-8b --persona formal_academic --baselines

    # Show specific method/layer/multiplier
    python scripts/inspect_outputs.py --model llama3.1-8b --persona formal_academic --method caa --layer 16 --multiplier 2.0

    # Show first N samples per condition
    python scripts/inspect_outputs.py --model llama3.1-8b --persona formal_academic -n 2
"""

import argparse
import json
import logging
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SEPARATOR = "-" * 80


def print_generations(data: dict, n: int):
    """Print generations from a single JSON result file."""
    generations = data["generations"][:n]
    for i, text in enumerate(generations):
        print(f"\n  [{i+1}] {text[:500]}{'...' if len(text) > 500 else ''}")


def inspect_baselines(gen_dir: Path, model: str, persona: str, n: int):
    """Show baseline (neutral + prompted) outputs."""
    for baseline in ["neutral", "prompted"]:
        path = gen_dir / model / persona / f"baseline_{baseline}.json"
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        print(f"\n{SEPARATOR}")
        print(f"BASELINE: {baseline.upper()}")
        print(f"Model: {model} | Persona: {persona}")
        print(SEPARATOR)
        print_generations(data, n)


def inspect_steered(
    gen_dir: Path,
    model: str,
    persona: str,
    n: int,
    method_filter: str | None = None,
    layer_filter: int | None = None,
    mult_filter: float | None = None,
):
    """Show steered outputs, optionally filtered."""
    methods_dir = gen_dir / model / persona
    if not methods_dir.exists():
        print(f"No generations found at {methods_dir}")
        return

    for method_dir in sorted(methods_dir.iterdir()):
        if not method_dir.is_dir() or method_dir.name.startswith("baseline"):
            continue
        method = method_dir.name
        if method_filter and method != method_filter:
            continue

        for gen_file in sorted(method_dir.glob("layer_*.json")):
            with open(gen_file) as f:
                data = json.load(f)

            layer = data["layer"]
            mult = data["multiplier"]

            if layer_filter is not None and layer != layer_filter:
                continue
            if mult_filter is not None and abs(mult - mult_filter) > 0.01:
                continue

            print(f"\n{SEPARATOR}")
            print(f"STEERED: {method.upper()} | Layer {layer} | Multiplier {mult}")
            print(f"Model: {model} | Persona: {persona}")
            print(SEPARATOR)
            print_generations(data, n)


def summarize(gen_dir: Path, model: str, persona: str):
    """Print a summary of what outputs exist."""
    base_dir = gen_dir / model / persona
    if not base_dir.exists():
        print(f"No outputs found at {base_dir}")
        return

    print(f"\n{SEPARATOR}")
    print(f"AVAILABLE OUTPUTS: {model} / {persona}")
    print(SEPARATOR)

    for baseline in ["neutral", "prompted"]:
        path = base_dir / f"baseline_{baseline}.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            print(f"  baseline_{baseline}: {len(data['generations'])} generations")

    for method_dir in sorted(base_dir.iterdir()):
        if not method_dir.is_dir() or method_dir.name.startswith("baseline"):
            continue
        files = sorted(method_dir.glob("layer_*.json"))
        if files:
            layers = set()
            mults = set()
            total = 0
            for f in files:
                with open(f) as fh:
                    d = json.load(fh)
                layers.add(d["layer"])
                mults.add(d["multiplier"])
                total += len(d["generations"])
            print(
                f"  {method_dir.name}: {len(files)} conditions, "
                f"{total} generations | "
                f"layers: {sorted(layers)} | "
                f"multipliers: {sorted(mults)}"
            )


def main():
    parser = argparse.ArgumentParser(description="Inspect steering outputs")
    parser.add_argument("--config", type=str, default="configs/dev.yaml")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--persona", type=str, required=True)
    parser.add_argument("-n", type=int, default=3, help="Samples to show per condition")
    parser.add_argument("--baselines", action="store_true", help="Show only baselines")
    parser.add_argument("--method", type=str, default=None, help="Filter by method")
    parser.add_argument("--layer", type=int, default=None, help="Filter by layer")
    parser.add_argument("--multiplier", type=float, default=None, help="Filter by multiplier")
    parser.add_argument("--summary", action="store_true", help="Show summary only")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    gen_dir = Path(config["project"]["output_dir"]) / "generations"

    if args.summary:
        summarize(gen_dir, args.model, args.persona)
        return

    if args.baselines:
        inspect_baselines(gen_dir, args.model, args.persona, args.n)
        return

    # Show baselines first, then steered
    inspect_baselines(gen_dir, args.model, args.persona, args.n)
    inspect_steered(
        gen_dir, args.model, args.persona, args.n,
        method_filter=args.method,
        layer_filter=args.layer,
        mult_filter=args.multiplier,
    )


if __name__ == "__main__":
    main()
