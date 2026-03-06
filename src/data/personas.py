"""Persona dataset generation for contrastive pairs and evaluation prompts.

Generates paired prompts for each persona using the model's own chat template,
ensuring consistent formatting across architectures.
"""

import logging
import random
from dataclasses import dataclass, field

import yaml

logger = logging.getLogger(__name__)

# Diverse evaluation prompts covering different tasks/domains
EVAL_PROMPTS = [
    "Explain how photosynthesis works.",
    "What are the main causes of climate change?",
    "How does a computer processor work?",
    "Describe the water cycle.",
    "What is the theory of evolution?",
    "Explain how vaccines work.",
    "What causes earthquakes?",
    "How does the stock market function?",
    "Describe the process of making bread.",
    "What is artificial intelligence?",
    "How do airplanes fly?",
    "Explain the concept of democracy.",
    "What is the role of DNA in living organisms?",
    "How does the internet work?",
    "Describe the solar system.",
    "What are the principles of good nutrition?",
    "How does a car engine work?",
    "Explain the concept of supply and demand.",
    "What is quantum mechanics?",
    "How do antibiotics fight infection?",
    "Describe the process of recycling.",
    "What causes rainbows?",
    "How does memory work in the human brain?",
    "Explain the greenhouse effect.",
    "What is the importance of biodiversity?",
    "How do social media algorithms work?",
    "Describe the history of the printing press.",
    "What is machine learning?",
    "How does the human immune system function?",
    "Explain the concept of compound interest.",
    "What are black holes?",
    "How does electricity reach our homes?",
    "Describe the process of fermentation.",
    "What is the significance of the Renaissance?",
    "How do tides work?",
    "Explain the scientific method.",
    "What causes seasons on Earth?",
    "How does GPS technology work?",
    "Describe the structure of an atom.",
    "What is the role of central banks?",
    "How do optical illusions work?",
    "Explain the concept of natural selection.",
    "What is cryptography?",
    "How does the human heart pump blood?",
    "Describe the water treatment process.",
    "What is game theory?",
    "How do satellites orbit Earth?",
    "Explain the Doppler effect.",
    "What are the fundamental forces of nature?",
    "How does a battery store energy?",
]


@dataclass
class PersonaConfig:
    """Configuration for a single persona."""
    id: str
    name: str
    description: str
    positive_system_prompt: str
    negative_system_prompt: str


@dataclass
class PersonaDataset:
    """Manages prompt generation for persona steering experiments."""
    personas: list[PersonaConfig] = field(default_factory=list)
    eval_prompts: list[str] = field(default_factory=lambda: EVAL_PROMPTS.copy())

    @classmethod
    def from_config(cls, config_path: str) -> "PersonaDataset":
        """Load personas from a YAML config file."""
        with open(config_path) as f:
            config = yaml.safe_load(f)

        personas = [PersonaConfig(**p) for p in config["personas"]]
        dataset = cls(personas=personas)
        logger.info(f"Loaded {len(personas)} personas from {config_path}")
        return dataset

    def get_contrastive_pairs(
        self,
        persona_id: str,
        tokenizer,
        num_pairs: int = 100,
        seed: int = 42,
    ) -> tuple[list[str], list[str]]:
        """Generate contrastive text pairs for a persona.

        Each pair consists of the same user prompt formatted with:
        - positive: the persona's system prompt
        - negative: the anti-persona system prompt

        Args:
            persona_id: Which persona to generate pairs for.
            tokenizer: HuggingFace tokenizer (for chat template).
            num_pairs: Number of contrastive pairs to generate.
            seed: Random seed for prompt selection.

        Returns:
            Tuple of (positive_texts, negative_texts).
        """
        persona = self._get_persona(persona_id)
        rng = random.Random(seed)
        prompts = rng.choices(self.eval_prompts, k=num_pairs)

        positive_texts = []
        negative_texts = []

        for user_prompt in prompts:
            pos_messages = [
                {"role": "system", "content": persona.positive_system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            neg_messages = [
                {"role": "system", "content": persona.negative_system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            positive_texts.append(
                tokenizer.apply_chat_template(
                    pos_messages, tokenize=False, add_generation_prompt=True
                )
            )
            negative_texts.append(
                tokenizer.apply_chat_template(
                    neg_messages, tokenize=False, add_generation_prompt=True
                )
            )

        logger.info(
            f"Generated {num_pairs} contrastive pairs for persona '{persona_id}'"
        )
        return positive_texts, negative_texts

    def get_eval_prompts(
        self,
        tokenizer,
        num_prompts: int = 50,
        seed: int = 42,
    ) -> list[str]:
        """Get formatted evaluation prompts (no system prompt).

        These are neutral prompts used to test steering effectiveness.
        """
        rng = random.Random(seed)
        prompts = rng.sample(
            self.eval_prompts, min(num_prompts, len(self.eval_prompts))
        )

        formatted = []
        for user_prompt in prompts:
            messages = [{"role": "user", "content": user_prompt}]
            formatted.append(
                tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )
        return formatted

    def get_prompted_texts(
        self,
        persona_id: str,
        tokenizer,
        num_prompts: int = 50,
        seed: int = 42,
    ) -> list[str]:
        """Get prompts with persona system prompt (for prompt engineering baseline)."""
        persona = self._get_persona(persona_id)
        rng = random.Random(seed)
        prompts = rng.sample(
            self.eval_prompts, min(num_prompts, len(self.eval_prompts))
        )

        formatted = []
        for user_prompt in prompts:
            messages = [
                {"role": "system", "content": persona.positive_system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            formatted.append(
                tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )
        return formatted

    def _get_persona(self, persona_id: str) -> PersonaConfig:
        """Look up a persona by ID."""
        for p in self.personas:
            if p.id == persona_id:
                return p
        raise ValueError(f"Unknown persona: {persona_id}")
