"""Phase 1: Extract steering vectors for all methods, models, and personas.

Usage:
    python scripts/run_extraction.py --config configs/default.yaml
    python scripts/run_extraction.py --config configs/default.yaml --model llama3.1-8b --persona formal_academic
"""

import argparse
import logging
from pathlib import Path

import yaml
from tqdm import tqdm

from src.extraction.hooks import ActivationExtractor
from src.extraction.caa import CAAExtractor
from src.extraction.pca import PCAExtractor
from src.extraction.linear_probe import LinearProbeExtractor
from src.data.personas import PersonaDataset
from src.utils.tracking import ExperimentTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def extract_vectors_for_model(
    model_cfg: dict,
    dataset: PersonaDataset,
    config: dict,
    tracker: ExperimentTracker,
    output_dir: Path,
    persona_filter: str | None = None,
):
    """Run all extraction methods for a single model."""
    model_name = model_cfg["name"]
    short_name = model_cfg["short_name"]
    ext_cfg = config["extraction"]

    logger.info(f"=== Processing model: {short_name} ===")

    # Load model once, reuse for all methods
    extractor = ActivationExtractor(model_name)
    extractor.load_model()

    caa = CAAExtractor(extractor)
    pca = PCAExtractor(extractor)
    probe = LinearProbeExtractor(
        extractor,
        epochs=config["evaluation"]["classifier"]["epochs"],
        lr=config["evaluation"]["classifier"]["lr"],
    )

    personas = dataset.personas
    if persona_filter:
        personas = [p for p in personas if p.id == persona_filter]

    for persona in tqdm(personas, desc=f"Personas ({short_name})"):
        logger.info(f"--- Persona: {persona.id} ---")

        pos_texts, neg_texts = dataset.get_contrastive_pairs(
            persona_id=persona.id,
            tokenizer=extractor.tokenizer,
            num_pairs=ext_cfg["num_pairs"],
            seed=config["project"]["seed"],
        )

        methods = {"caa": caa, "pca": pca, "linear_probe": probe}
        for method_name, method in methods.items():
            if method_name not in ext_cfg["methods"]:
                continue

            logger.info(f"Extracting {method_name} vectors...")
            vectors = method.extract_vectors(
                positive_texts=pos_texts,
                negative_texts=neg_texts,
                token_position=ext_cfg["token_position"],
                batch_size=ext_cfg["batch_size"],
            )

            # Save vectors
            vec_path = output_dir / short_name / persona.id / f"{method_name}_vectors.pt"
            method.save_vectors(vectors, vec_path)

            # Log vector norms per layer
            for layer_idx, vec in vectors.items():
                tracker.log_metrics({
                    f"vector_norm/{short_name}/{persona.id}/{method_name}/layer_{layer_idx}": vec.norm().item(),
                })

    # Clean up model from memory
    del extractor.model
    del extractor


def main():
    parser = argparse.ArgumentParser(description="Extract steering vectors")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model", type=str, default=None, help="Filter to single model (short_name)")
    parser.add_argument("--persona", type=str, default=None, help="Filter to single persona")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["project"]["output_dir"]) / "vectors"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = PersonaDataset.from_config(args.config)
    tracker = ExperimentTracker(config)

    models = config["models"]
    if args.model:
        models = [m for m in models if m["short_name"] == args.model]

    for model_cfg in models:
        extract_vectors_for_model(
            model_cfg=model_cfg,
            dataset=dataset,
            config=config,
            tracker=tracker,
            output_dir=output_dir,
            persona_filter=args.persona,
        )

    tracker.finish()
    logger.info("Extraction complete.")


if __name__ == "__main__":
    main()
