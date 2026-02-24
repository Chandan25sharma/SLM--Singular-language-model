"""
SLM Engine - Pure Python Implementation (Fallback)
===================================================
This is the pure-Python n-gram engine. It works identically to the C++ engine
but runs without needing a compiled shared library. The C++ engine is ~10-100x
faster for large corpora, but this works out-of-the-box.
"""

import os
import re
import json
import math
import struct
import random
import pickle
from collections import defaultdict
from typing import List, Tuple, Dict, Optional


class Tokenizer:
    """Splits text into lowercase word tokens, strips punctuation."""

    @staticmethod
    def tokenize(text: str) -> List[str]:
        text = text.lower()
        # Keep apostrophes within words, strip all other punctuation
        words = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text)
        return words

    @staticmethod
    def detokenize(tokens: List[str]) -> str:
        return " ".join(tokens)


class FrequencyStore:
    """Hash-map storage for n-gram frequency counts."""

    def __init__(self):
        self._store: Dict[str, int] = defaultdict(int)

    def increment(self, key: str, count: int = 1):
        self._store[key] += count

    def get_count(self, key: str) -> int:
        return self._store.get(key, 0)

    def get_top_k(self, prefix: str, k: int) -> List[Tuple[str, int]]:
        matches = [
            (key, count)
            for key, count in self._store.items()
            if key.startswith(prefix)
        ]
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:k]

    def size(self) -> int:
        return len(self._store)

    def total_count(self) -> int:
        return sum(self._store.values())

    def clear(self):
        self._store.clear()

    @property
    def entries(self) -> Dict[str, int]:
        return dict(self._store)


class NGramModel:
    """N-gram frequency model with unigram/bigram/trigram support and backoff."""

    def __init__(self):
        self.unigrams = FrequencyStore()
        self.bigrams = FrequencyStore()
        self.trigrams = FrequencyStore()
        self.total_tokens = 0

    @staticmethod
    def _bigram_key(w1: str, w2: str) -> str:
        return f"{w1}|{w2}"

    @staticmethod
    def _trigram_key(w1: str, w2: str, w3: str) -> str:
        return f"{w1}|{w2}|{w3}"

    @staticmethod
    def _bigram_prefix(w1: str) -> str:
        return f"{w1}|"

    @staticmethod
    def _trigram_prefix(w1: str, w2: str) -> str:
        return f"{w1}|{w2}|"

    @staticmethod
    def _extract_last_word(key: str) -> str:
        pos = key.rfind("|")
        return key[pos + 1:] if pos != -1 else key

    def train(self, tokens: List[str]):
        """Train on a sequence of tokens."""
        if not tokens:
            return

        self.total_tokens += len(tokens)

        # Unigrams
        for token in tokens:
            self.unigrams.increment(token)

        # Bigrams
        for i in range(len(tokens) - 1):
            self.bigrams.increment(self._bigram_key(tokens[i], tokens[i + 1]))

        # Trigrams
        for i in range(len(tokens) - 2):
            self.trigrams.increment(
                self._trigram_key(tokens[i], tokens[i + 1], tokens[i + 2])
            )

    def predict_next(
        self, context: List[str], num_candidates: int = 5
    ) -> List[Tuple[str, float]]:
        """Predict next word using trigram -> bigram -> unigram backoff."""
        results = []

        # Try trigram
        if len(context) >= 2:
            w1, w2 = context[-2], context[-1]
            prefix = self._trigram_prefix(w1, w2)
            top = self.trigrams.get_top_k(prefix, num_candidates * 2)

            if top:
                context_count = self.bigrams.get_count(self._bigram_key(w1, w2))
                if context_count > 0:
                    for key, count in top:
                        word = self._extract_last_word(key)
                        prob = count / context_count
                        results.append((word, prob))

        # Backoff to bigram
        if not results and context:
            w1 = context[-1]
            prefix = self._bigram_prefix(w1)
            top = self.bigrams.get_top_k(prefix, num_candidates * 2)

            if top:
                context_count = self.unigrams.get_count(w1)
                if context_count > 0:
                    for key, count in top:
                        word = self._extract_last_word(key)
                        prob = count / context_count
                        results.append((word, prob))

        # Backoff to unigram
        if not results:
            top = self.unigrams.get_top_k("", num_candidates * 2)
            total = self.unigrams.total_count()
            if total > 0:
                for key, count in top:
                    prob = count / total
                    results.append((key, prob))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:num_candidates]

    def save_model(self, path: str) -> bool:
        """Save model to disk using pickle."""
        try:
            data = {
                "version": "SLM1",
                "total_tokens": self.total_tokens,
                "unigrams": self.unigrams.entries,
                "bigrams": self.bigrams.entries,
                "trigrams": self.trigrams.entries,
            }
            with open(path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            return True
        except Exception:
            return False

    def load_model(self, path: str) -> bool:
        """Load model from disk."""
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)

            if data.get("version") != "SLM1":
                return False

            self.total_tokens = data["total_tokens"]
            self.unigrams = FrequencyStore()
            self.bigrams = FrequencyStore()
            self.trigrams = FrequencyStore()

            for key, count in data["unigrams"].items():
                self.unigrams.increment(key, count)
            for key, count in data["bigrams"].items():
                self.bigrams.increment(key, count)
            for key, count in data["trigrams"].items():
                self.trigrams.increment(key, count)

            return True
        except Exception:
            return False

    def vocab_size(self) -> int:
        return self.unigrams.size()

    def top_vocab(self, n: int) -> List[Tuple[str, int]]:
        return self.unigrams.get_top_k("", n)

    def reset(self):
        self.unigrams.clear()
        self.bigrams.clear()
        self.trigrams.clear()
        self.total_tokens = 0


class Generator:
    """Text generator using n-gram probabilities with temperature sampling."""

    def __init__(self, model: NGramModel):
        self.model = model

    def sample(
        self, candidates: List[Tuple[str, float]], temperature: float
    ) -> str:
        """Weighted random selection with temperature scaling."""
        if not candidates:
            return ""
        if len(candidates) == 1:
            return candidates[0][0]

        # Apply temperature
        weights = []
        for word, prob in candidates:
            adjusted = prob ** (1.0 / temperature) if temperature > 0 else prob
            weights.append(adjusted)

        # Normalize
        total = sum(weights)
        if total <= 0:
            return candidates[0][0]

        weights = [w / total for w in weights]

        # Weighted random choice
        r = random.random()
        cumulative = 0.0
        for i, w in enumerate(weights):
            cumulative += w
            if r <= cumulative:
                return candidates[i][0]

        return candidates[-1][0]

    def generate(
        self,
        prompt_tokens: List[str],
        max_length: int = 50,
        temperature: float = 0.7,
    ) -> List[str]:
        """Generate text continuation from prompt tokens."""
        result = list(prompt_tokens)

        for i in range(max_length):
            # Build context from last 2 tokens
            context = result[-2:] if len(result) >= 2 else result[-1:] if result else []

            candidates = self.model.predict_next(context, 10)
            if not candidates:
                break

            next_word = self.sample(candidates, temperature)
            if not next_word:
                break

            result.append(next_word)

            # Natural stopping heuristic
            if i > 15:
                stop_chance = (i - 15) / (max_length * 2.0)
                if random.random() < stop_chance:
                    break

        return result


class SLMEngine:
    """
    Main SLM Engine — wraps the n-gram model, tokenizer, and generator.
    This is the pure-Python implementation (no C++ dependency).
    """

    def __init__(self):
        self.model = NGramModel()
        self._trained = False

    def train_text(self, text: str) -> int:
        """Train on raw text. Returns number of tokens processed."""
        tokens = Tokenizer.tokenize(text)
        if not tokens:
            return 0
        self.model.train(tokens)
        self._trained = True
        return len(tokens)

    def train_file(self, filepath: str) -> int:
        """Train on a text file. Returns number of tokens processed."""
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return self.train_text(text)

    def train_directory(self, dirpath: str) -> int:
        """Train on all .txt files in a directory. Returns total tokens."""
        total = 0
        for filename in sorted(os.listdir(dirpath)):
            if filename.endswith(".txt"):
                filepath = os.path.join(dirpath, filename)
                count = self.train_file(filepath)
                total += count
        return total

    def generate(
        self, prompt: str, max_length: int = 50, temperature: float = 0.7
    ) -> str:
        """Generate text from a prompt."""
        tokens = Tokenizer.tokenize(prompt)
        gen = Generator(self.model)
        result = gen.generate(tokens, max_length, temperature)
        return Tokenizer.detokenize(result)

    def predict_next(
        self, context: str, num_candidates: int = 5
    ) -> List[Tuple[str, float]]:
        """Get next-word predictions for a context."""
        tokens = Tokenizer.tokenize(context)
        return self.model.predict_next(tokens, num_candidates)

    def save_model(self, path: str) -> bool:
        return self.model.save_model(path)

    def load_model(self, path: str) -> bool:
        success = self.model.load_model(path)
        if success:
            self._trained = True
        return success

    def vocab_size(self) -> int:
        return self.model.vocab_size()

    def total_tokens(self) -> int:
        return self.model.total_tokens

    def top_vocab(self, n: int = 20) -> List[Tuple[str, int]]:
        return self.model.top_vocab(n)

    def reset(self):
        self.model.reset()
        self._trained = False

    @property
    def is_trained(self) -> bool:
        return self._trained
