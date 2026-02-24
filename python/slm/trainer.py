"""
SLM Trainer - Training pipeline for SLM models.
"""

import os
import sys
import time
from typing import Optional
from .engine import SLMEngine


class SLMTrainer:
    """Training pipeline with progress reporting."""

    def __init__(self, engine: SLMEngine):
        self.engine = engine

    def train_file(self, filepath: str, verbose: bool = True) -> int:
        """Train on a single text file with progress output."""
        if not os.path.exists(filepath):
            print(f"\033[91m  ERROR: File not found: {filepath}\033[0m")
            return 0

        filename = os.path.basename(filepath)
        filesize = os.path.getsize(filepath)

        if verbose:
            print(f"\033[36m  Training on: {filename} ({filesize:,} bytes)\033[0m")

        start = time.time()
        tokens = self.engine.train_file(filepath)
        elapsed = time.time() - start

        if verbose:
            rate = tokens / elapsed if elapsed > 0 else 0
            print(f"\033[32m  ✓ Processed {tokens:,} tokens in {elapsed:.2f}s ({rate:,.0f} tokens/sec)\033[0m")

        return tokens

    def train_directory(self, dirpath: str, verbose: bool = True) -> int:
        """Train on all .txt files in a directory."""
        if not os.path.isdir(dirpath):
            print(f"\033[91m  ERROR: Directory not found: {dirpath}\033[0m")
            return 0

        txt_files = sorted([
            f for f in os.listdir(dirpath)
            if f.endswith(".txt")
        ])

        if not txt_files:
            print(f"\033[93m  WARNING: No .txt files found in {dirpath}\033[0m")
            return 0

        if verbose:
            print(f"\033[36m  Found {len(txt_files)} text file(s) in {dirpath}\033[0m")

        total_tokens = 0
        start = time.time()

        for i, filename in enumerate(txt_files, 1):
            filepath = os.path.join(dirpath, filename)
            if verbose:
                print(f"\033[90m  [{i}/{len(txt_files)}] {filename}\033[0m", end=" ")

            tokens = self.engine.train_file(filepath)
            total_tokens += tokens

            if verbose:
                print(f"\033[32m({tokens:,} tokens)\033[0m")

        elapsed = time.time() - start

        if verbose:
            print(f"\n\033[32m  ✓ Total: {total_tokens:,} tokens from {len(txt_files)} files in {elapsed:.2f}s\033[0m")

        return total_tokens

    def train_text(self, text: str, label: str = "inline", verbose: bool = True) -> int:
        """Train on raw text input."""
        if verbose:
            print(f"\033[36m  Training on {label} ({len(text):,} chars)\033[0m")

        start = time.time()
        tokens = self.engine.train_text(text)
        elapsed = time.time() - start

        if verbose:
            print(f"\033[32m  ✓ Processed {tokens:,} tokens in {elapsed:.2f}s\033[0m")

        return tokens
