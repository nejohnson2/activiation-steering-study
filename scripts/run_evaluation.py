"""Phase 4: Evaluate all generated outputs.

Runs the three-tier evaluation: representation metrics, persona classifier, and LLM judge.

Usage:
    python scripts/run_evaluation.py --config configs/default.yaml
    python scripts/run_evaluation.py --config configs/default.yaml --tier classifier
    python scripts/run_evaluation.py --config configs/default.yaml --tier judge
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from src.evaluation.classifier import PersonaClassifier
from src.evaluation.llm_judge import LLMJudge
from src.data.personas import PersonaDataset
from src.utils.tracking import ExperimentTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def train_persona_classifier(
    dataset: PersonaDataset,
    generations_dir: Path,
    config: dict,
) -> PersonaClassifier:
    """Train classifier on prompted baseline outputs (synthetic persona data)."""
    persona_ids = [p.id for p in dataset.personas]
    classifier = PersonaClassifier(
        persona_ids=persona_ids,
        hidden_dim=config["evaluation"]["classifier"]["hidden_dim"],
    )

    training_data = {}
    for model_cfg in config["models"]:
        short_name = model_cfg["short_name"]
        for persona in dataset.personas:
            gen_path = generations_dir / short_name / persona.id / "baseline_prompted.json"
            if gen_path.exists():
                with open(gen_path) as f:
                    data = json.load(f)
                if persona.id not in training_data:
                    training_data[persona.id] = []
                training_data[persona.id].extend(data["generations"])

    if not training_data:
        logger.error("No prompted baseline data found for classifier training")
        return classifier

    accuracy = classifier.train_classifier(
        training_data=training_data,
        epochs=config["evaluation"]["classifier"]["epochs"],
        lr=config["evaluation"]["classifier"]["lr"],
    )
    logger.info(f"Persona classifier trained with accuracy: {accuracy:.3f}")
    return classifier


def evaluate_with_classifier(
    classifier: PersonaClassifier,
    generations_dir: Path,
    config: dict,
    tracker: ExperimentTracker,
    output_dir: Path,
):
    """Run classifier evaluation on all generations."""
    all_results = []

    for model_cfg in config["models"]:
        short_name = model_cfg["short_name"]
        for persona_cfg in config["personas"]:
            persona_id = persona_cfg["id"]

            # Evaluate baselines
            for baseline in ["neutral", "prompted"]:
                gen_path = generations_dir / short_name / persona_id / f"baseline_{baseline}.json"
                if not gen_path.exists():
                    continue
                with open(gen_path) as f:
                    data = json.load(f)
                score = classifier.score(data["generations"], persona_id)
                result = {
                    "model": short_name,
                    "persona": persona_id,
                    "condition": f"baseline_{baseline}",
                    "classifier_score": score,
                }
                all_results.append(result)
                tracker.log_metrics({
                    f"classifier/{short_name}/{persona_id}/baseline_{baseline}": score
                })

            # Evaluate steered generations
            for method in config["extraction"]["methods"]:
                method_dir = generations_dir / short_name / persona_id / method
                if not method_dir.exists():
                    continue
                for gen_file in sorted(method_dir.glob("layer_*.json")):
                    with open(gen_file) as f:
                        data = json.load(f)
                    score = classifier.score(data["generations"], persona_id)
                    result = {
                        "model": short_name,
                        "persona": persona_id,
                        "method": method,
                        "layer": data["layer"],
                        "multiplier": data["multiplier"],
                        "condition": f"{method}_L{data['layer']}_M{data['multiplier']}",
                        "classifier_score": score,
                    }
                    all_results.append(result)
                    tracker.log_metrics({
                        f"classifier/{short_name}/{persona_id}/{method}/L{data['layer']}_M{data['multiplier']}": score
                    })

    results_path = output_dir / "classifier_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Classifier results saved to {results_path}")


def evaluate_with_judge(
    dataset: PersonaDataset,
    generations_dir: Path,
    config: dict,
    tracker: ExperimentTracker,
    output_dir: Path,
):
    """Run LLM judge evaluation on a subset of generations."""
    judge_cfg = config["evaluation"]["llm_judge"]
    judge = LLMJudge(
        model=judge_cfg["model"],
        base_url=judge_cfg.get("base_url", "http://localhost:11434"),
    )
    num_samples = judge_cfg["num_judge_samples"]
    all_results = []

    for model_cfg in config["models"]:
        short_name = model_cfg["short_name"]
        for persona in dataset.personas:
            # Judge baselines
            for baseline in ["neutral", "prompted"]:
                gen_path = generations_dir / short_name / persona.id / f"baseline_{baseline}.json"
                if not gen_path.exists():
                    continue
                with open(gen_path) as f:
                    data = json.load(f)
                texts = data["generations"][:num_samples]
                results = judge.evaluate_batch(
                    texts=texts,
                    persona_name=persona.name,
                    persona_description=persona.description,
                    persona_id=persona.id,
                )
                agg = LLMJudge.aggregate_scores(results)
                all_results.append({
                    "model": short_name,
                    "persona": persona.id,
                    "condition": f"baseline_{baseline}",
                    **agg,
                })
                tracker.log_metrics({
                    f"judge/{short_name}/{persona.id}/baseline_{baseline}": agg["mean"]
                })

            # Judge steered (best layer per method based on classifier scores if available)
            for method in config["extraction"]["methods"]:
                method_dir = generations_dir / short_name / persona.id / method
                if not method_dir.exists():
                    continue
                for gen_file in sorted(method_dir.glob("layer_*.json")):
                    with open(gen_file) as f:
                        data = json.load(f)
                    texts = data["generations"][:num_samples]
                    results = judge.evaluate_batch(
                        texts=texts,
                        persona_name=persona.name,
                        persona_description=persona.description,
                        persona_id=persona.id,
                    )
                    agg = LLMJudge.aggregate_scores(results)
                    all_results.append({
                        "model": short_name,
                        "persona": persona.id,
                        "method": method,
                        "layer": data["layer"],
                        "multiplier": data["multiplier"],
                        "condition": f"{method}_L{data['layer']}_M{data['multiplier']}",
                        **agg,
                    })
                    tracker.log_metrics({
                        f"judge/{short_name}/{persona.id}/{method}/L{data['layer']}_M{data['multiplier']}": agg["mean"]
                    })

    results_path = output_dir / "judge_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Judge results saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate steering results")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--tier", type=str, default="all", choices=["all", "classifier", "judge"])
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    generations_dir = Path(config["project"]["output_dir"]) / "generations"
    output_dir = Path(config["project"]["output_dir"]) / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = PersonaDataset.from_config(args.config)
    tracker = ExperimentTracker(config)

    if args.tier in ("all", "classifier"):
        logger.info("=== Training persona classifier ===")
        classifier = train_persona_classifier(dataset, generations_dir, config)
        classifier.save(output_dir / "persona_classifier.pt")

        logger.info("=== Running classifier evaluation ===")
        evaluate_with_classifier(classifier, generations_dir, config, tracker, output_dir)

    if args.tier in ("all", "judge"):
        logger.info("=== Running LLM judge evaluation ===")
        evaluate_with_judge(dataset, generations_dir, config, tracker, output_dir)

    tracker.finish()
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
