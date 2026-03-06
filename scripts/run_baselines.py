"""Phase 3: Generate prompt engineering baselines.

Generates outputs using system-prompt-based persona control (no steering).
This provides the comparison baseline for RQ3.

Usage:
    python scripts/run_baselines.py --config configs/default.yaml
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.personas import PersonaDataset
from src.utils.device import get_device, get_dtype
from src.utils.tracking import ExperimentTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_baselines_for_model(
    model_cfg: dict,
    dataset: PersonaDataset,
    config: dict,
    tracker: ExperimentTracker,
    output_dir: Path,
    persona_filter: str | None = None,
):
    """Generate prompt engineering baselines for a single model."""
    model_name = model_cfg["name"]
    short_name = model_cfg["short_name"]
    steer_cfg = config["steering"]

    device = get_device()
    dtype = get_dtype(device)

    logger.info(f"=== Prompt baselines: {short_name} ===")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    if device.type != "cuda":
        model = model.to(device)
    model.eval()

    personas = dataset.personas
    if persona_filter:
        personas = [p for p in personas if p.id == persona_filter]

    for persona in personas:
        logger.info(f"--- Persona: {persona.id} ---")

        # Baseline 1: No persona (neutral)
        neutral_prompts = dataset.get_eval_prompts(
            tokenizer=tokenizer,
            num_prompts=steer_cfg["num_eval_prompts"],
            seed=config["project"]["seed"],
        )

        # Baseline 2: System prompt persona
        prompted_texts = dataset.get_prompted_texts(
            persona_id=persona.id,
            tokenizer=tokenizer,
            num_prompts=steer_cfg["num_eval_prompts"],
            seed=config["project"]["seed"],
        )

        for baseline_name, prompts in [("neutral", neutral_prompts), ("prompted", prompted_texts)]:
            generations = []
            for prompt in tqdm(prompts, desc=f"{baseline_name}"):
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=steer_cfg["max_gen_tokens"],
                        temperature=steer_cfg["temperature"],
                        top_p=steer_cfg["top_p"],
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                generated = outputs[0, inputs["input_ids"].shape[1]:]
                text = tokenizer.decode(generated, skip_special_tokens=True)
                generations.append(text)

            gen_path = output_dir / short_name / persona.id / f"baseline_{baseline_name}.json"
            gen_path.parent.mkdir(parents=True, exist_ok=True)
            with open(gen_path, "w") as f:
                json.dump({
                    "model": short_name,
                    "persona": persona.id,
                    "baseline": baseline_name,
                    "generations": generations,
                }, f, indent=2)

            tracker.log_metrics({
                f"baseline_count/{short_name}/{persona.id}/{baseline_name}": len(generations),
            })

    del model


def main():
    parser = argparse.ArgumentParser(description="Run prompt engineering baselines")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--persona", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["project"]["output_dir"]) / "generations"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = PersonaDataset.from_config(args.config)
    tracker = ExperimentTracker(config)

    models = config["models"]
    if args.model:
        models = [m for m in models if m["short_name"] == args.model]

    for model_cfg in models:
        run_baselines_for_model(
            model_cfg=model_cfg,
            dataset=dataset,
            config=config,
            tracker=tracker,
            output_dir=output_dir,
            persona_filter=args.persona,
        )

    tracker.finish()
    logger.info("Baselines complete.")


if __name__ == "__main__":
    main()
