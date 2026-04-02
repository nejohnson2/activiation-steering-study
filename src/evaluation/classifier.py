"""Persona classifier for automated output evaluation.

Trained on synthetic data generated via strong prompt engineering, then used
to score whether steered outputs exhibit the target persona.
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

logger = logging.getLogger(__name__)


class PersonaClassifierModel(nn.Module):
    """Simple MLP classifier on top of sentence embeddings."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PersonaClassifier:
    """Train and use a persona classifier for output evaluation.

    Uses a small sentence-transformer to embed generated texts, then classifies
    them into persona categories. Trained entirely on synthetic data.
    """

    def __init__(
        self,
        persona_ids: list[str],
        embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        hidden_dim: int = 256,
        device: torch.device | None = None,
    ):
        self.persona_ids = persona_ids
        self.id_to_idx = {pid: i for i, pid in enumerate(persona_ids)}
        self.embed_model_name = embed_model_name
        self.hidden_dim = hidden_dim
        self.device = device or torch.device("cpu")
        self.embed_model = None
        self.embed_tokenizer = None
        self.classifier = None

    def _load_embedder(self):
        """Load the sentence embedding model."""
        if self.embed_model is None:
            self.embed_tokenizer = AutoTokenizer.from_pretrained(self.embed_model_name)
            self.embed_model = AutoModel.from_pretrained(self.embed_model_name).to(self.device)
            self.embed_model.eval()

    def _embed_texts(self, texts: list[str], batch_size: int = 32) -> torch.Tensor:
        """Compute sentence embeddings via mean pooling."""
        self._load_embedder()
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.embed_tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = self.embed_model(**inputs)
                # Mean pooling over non-padding tokens
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                embeddings = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
                all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)

    def train_classifier(
        self,
        training_data: dict[str, list[str]],
        epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 32,
    ) -> float:
        """Train the persona classifier on synthetic labeled data.

        Args:
            training_data: Dict mapping persona_id -> list of example texts.
            epochs: Training epochs.
            lr: Learning rate.
            batch_size: Training batch size.

        Returns:
            Final training accuracy.
        """
        logger.info("Embedding training data for persona classifier...")
        all_embeddings = []
        all_labels = []

        for persona_id, texts in training_data.items():
            embeddings = self._embed_texts(texts, batch_size=batch_size)
            labels = torch.full((len(texts),), self.id_to_idx[persona_id], dtype=torch.long)
            all_embeddings.append(embeddings)
            all_labels.append(labels)

        X = torch.cat(all_embeddings, dim=0)
        y = torch.cat(all_labels, dim=0)

        input_dim = X.shape[1]
        self.classifier = PersonaClassifierModel(
            input_dim, self.hidden_dim, len(self.persona_ids)
        ).to(self.device)

        dataset = TensorDataset(X.to(self.device), y.to(self.device))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        self.classifier.train()
        for epoch in tqdm(range(epochs), desc="Training classifier"):
            total_loss = 0
            correct = 0
            total = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                logits = self.classifier(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                correct += (logits.argmax(dim=1) == batch_y).sum().item()
                total += batch_y.shape[0]

            if (epoch + 1) % 5 == 0:
                acc = correct / total
                logger.info(f"Epoch {epoch+1}: loss={total_loss/len(loader):.4f}, acc={acc:.3f}")

        final_acc = correct / total
        logger.info(f"Classifier training complete. Final accuracy: {final_acc:.3f}")
        return final_acc

    def predict(self, texts: list[str], batch_size: int = 32) -> dict[str, list[float]]:
        """Predict persona probabilities for a list of texts.

        Returns:
            Dict mapping persona_id -> list of probabilities (one per text).
        """
        self.classifier.eval()
        embeddings = self._embed_texts(texts, batch_size=batch_size).to(self.device)

        with torch.no_grad():
            logits = self.classifier(embeddings)
            probs = torch.softmax(logits, dim=1).cpu()

        result = {}
        for persona_id, idx in self.id_to_idx.items():
            result[persona_id] = probs[:, idx].tolist()
        return result

    def score(self, texts: list[str], target_persona: str, batch_size: int = 32) -> float:
        """Score how well texts match a target persona.

        Returns:
            Mean probability of target persona across all texts.
        """
        probs = self.predict(texts, batch_size=batch_size)
        return sum(probs[target_persona]) / len(probs[target_persona])

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Get embed_dim from classifier input layer
        embed_dim = self.classifier.net[0].in_features
        torch.save({
            "classifier_state": self.classifier.state_dict(),
            "persona_ids": self.persona_ids,
            "hidden_dim": self.hidden_dim,
            "embed_model_name": self.embed_model_name,
            "embed_dim": embed_dim,
        }, path)
        logger.info(f"Saved persona classifier to {path}")

    @classmethod
    def load(cls, path: str | Path, device: torch.device | None = None) -> "PersonaClassifier":
        data = torch.load(path, weights_only=False)
        obj = cls(
            persona_ids=data["persona_ids"],
            embed_model_name=data["embed_model_name"],
            hidden_dim=data["hidden_dim"],
            device=device or torch.device("cpu"),
        )
        # Reconstruct classifier architecture
        input_dim = data["embed_dim"]
        obj.classifier = PersonaClassifierModel(
            input_dim, data["hidden_dim"], len(data["persona_ids"])
        ).to(obj.device)
        obj.classifier.load_state_dict(data["classifier_state"])
        return obj
