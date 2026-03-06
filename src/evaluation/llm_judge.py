"""LLM-as-judge evaluation for persona adherence.

Uses a local LLM via Ollama to rate how well generated outputs
match a target persona on a Likert scale.
"""

import json
import logging
import statistics
from dataclasses import dataclass

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

JUDGE_PROMPT_TEMPLATE = """You are evaluating whether a text exhibits a specific persona.

**Target Persona:** {persona_name}
**Persona Description:** {persona_description}

**Text to evaluate:**
{text}

Rate how strongly the text exhibits the target persona on a scale of 1-5:
1 = No evidence of the persona at all
2 = Minimal hints of the persona
3 = Moderate persona presence, but inconsistent
4 = Strong persona presence throughout most of the text
5 = Perfect persona embodiment, consistent and unmistakable

Respond with ONLY a JSON object in this exact format:
{{"score": <int 1-5>, "reasoning": "<brief explanation>"}}"""


@dataclass
class JudgeResult:
    """Result from a single LLM judge evaluation."""
    score: int
    reasoning: str
    persona_id: str
    text: str


class LLMJudge:
    """Evaluate persona adherence using a local Ollama model as judge."""

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._verify_connection()

    def _verify_connection(self):
        """Check that Ollama is running and the model is available."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            available = [m["name"] for m in resp.json().get("models", [])]
            # Check for exact match or prefix match (e.g. "llama3.1:8b" matches "llama3.1:8b-instruct-...")
            model_base = self.model.split(":")[0]
            if not any(model_base in m for m in available):
                logger.warning(
                    f"Model '{self.model}' not found in Ollama. "
                    f"Available: {available}. Run: ollama pull {self.model}"
                )
            else:
                logger.info(f"Ollama connected. Using model: {self.model}")
        except requests.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running: ollama serve"
            )

    def evaluate_single(
        self,
        text: str,
        persona_name: str,
        persona_description: str,
        persona_id: str,
    ) -> JudgeResult:
        """Evaluate a single text for persona adherence."""
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            persona_name=persona_name,
            persona_description=persona_description,
            text=text,
        )

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 256,
                },
                "format": "json",
            },
            timeout=120,
        )
        response.raise_for_status()
        response_text = response.json()["response"].strip()

        try:
            parsed = json.loads(response_text)
            score = int(parsed["score"])
            score = max(1, min(5, score))  # clamp to valid range
            reasoning = parsed.get("reasoning", "")
        except (json.JSONDecodeError, KeyError, ValueError):
            logger.warning(f"Failed to parse judge response: {response_text}")
            score = 0
            reasoning = f"Parse error: {response_text}"

        return JudgeResult(
            score=score,
            reasoning=reasoning,
            persona_id=persona_id,
            text=text,
        )

    def evaluate_batch(
        self,
        texts: list[str],
        persona_name: str,
        persona_description: str,
        persona_id: str,
    ) -> list[JudgeResult]:
        """Evaluate a batch of texts for persona adherence."""
        results = []
        for text in tqdm(texts, desc=f"Judging {persona_id}"):
            result = self.evaluate_single(
                text=text,
                persona_name=persona_name,
                persona_description=persona_description,
                persona_id=persona_id,
            )
            results.append(result)
        return results

    @staticmethod
    def aggregate_scores(results: list[JudgeResult]) -> dict[str, float]:
        """Compute aggregate statistics from judge results."""
        valid_scores = [r.score for r in results if r.score > 0]
        if not valid_scores:
            return {"mean": 0.0, "std": 0.0, "n_valid": 0, "n_total": len(results)}

        return {
            "mean": statistics.mean(valid_scores),
            "std": statistics.stdev(valid_scores) if len(valid_scores) > 1 else 0.0,
            "n_valid": len(valid_scores),
            "n_total": len(results),
        }
