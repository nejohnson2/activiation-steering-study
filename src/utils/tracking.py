"""Experiment tracking with W&B fallback to file-based logging."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Unified experiment tracker supporting W&B and file-based backends."""

    def __init__(self, config: dict):
        self.config = config
        tracking_cfg = config.get("tracking", {})
        self.backend = tracking_cfg.get("backend", "file")
        self._run = None

        if self.backend == "wandb":
            try:
                import wandb
                wandb_cfg = tracking_cfg.get("wandb", {})
                self._run = wandb.init(
                    project=wandb_cfg.get("project", "activation-steering-study"),
                    entity=wandb_cfg.get("entity"),
                    config=config,
                )
                logger.info(f"W&B run initialized: {self._run.url}")
            except Exception as e:
                logger.warning(f"W&B init failed ({e}), falling back to file-based tracking")
                self.backend = "file"

        if self.backend == "file":
            file_cfg = tracking_cfg.get("file", {})
            self.log_dir = Path(file_cfg.get("log_dir", "results/logs"))
            self.log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = self.log_dir / f"run_{timestamp}.jsonl"
            self._write_entry({"type": "config", "data": config})
            logger.info(f"File-based tracking: {self.log_file}")

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None):
        """Log metrics to the active backend."""
        if self.backend == "wandb" and self._run is not None:
            import wandb
            wandb.log(metrics, step=step)
        else:
            self._write_entry({"type": "metrics", "step": step, "data": metrics})

    def log_artifact(self, name: str, path: str, artifact_type: str = "result"):
        """Log an artifact (file) to the active backend."""
        if self.backend == "wandb" and self._run is not None:
            import wandb
            artifact = wandb.Artifact(name, type=artifact_type)
            artifact.add_file(path)
            self._run.log_artifact(artifact)
        else:
            self._write_entry({
                "type": "artifact",
                "name": name,
                "path": path,
                "artifact_type": artifact_type,
            })

    def log_summary(self, summary: dict[str, Any]):
        """Log summary metrics."""
        if self.backend == "wandb" and self._run is not None:
            for k, v in summary.items():
                self._run.summary[k] = v
        else:
            self._write_entry({"type": "summary", "data": summary})

    def finish(self):
        """Finalize the tracking run."""
        if self.backend == "wandb" and self._run is not None:
            self._run.finish()
        logger.info("Experiment tracking finished")

    def _write_entry(self, entry: dict):
        """Write a JSON entry to the log file."""
        entry["timestamp"] = datetime.now().isoformat()
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
