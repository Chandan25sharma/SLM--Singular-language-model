"""
SLM Commands - Command registry and handlers for the SLM CLI.
Each command is a function that takes (engine, trainer, args).
"""

import os
import sys
import time
from typing import List, Optional
from .engine import SLMEngine
from .trainer import SLMTrainer
from .bridge import get_engine_type
from . import __version__


# ─── ANSI Colors ────────────────────────────────────────────────────────────────
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"
    GRAY    = "\033[90m"


# ─── Command Registry ───────────────────────────────────────────────────────────
COMMANDS = {}


def command(name: str, usage: str, description: str):
    """Decorator to register a command."""
    def decorator(func):
        COMMANDS[name] = {
            "func": func,
            "usage": usage,
            "description": description,
        }
        return func
    return decorator


# ─── Commands ────────────────────────────────────────────────────────────────────

@command("train", "train <file|directory>", "Train model on text file(s)")
def cmd_train(engine: SLMEngine, trainer: SLMTrainer, args: List[str]):
    if not args:
        print(f"{C.RED}  Usage: train <file_or_directory>{C.RESET}")
        return

    target = args[0]

    if os.path.isfile(target):
        trainer.train_file(target)
    elif os.path.isdir(target):
        trainer.train_directory(target)
    else:
        print(f"{C.RED}  ERROR: '{target}' not found{C.RESET}")
        return

    print(f"{C.CYAN}  Model: {engine.vocab_size():,} unique words, {engine.total_tokens():,} total tokens{C.RESET}")


@command("train-text", 'train-text "<text>"', "Train on inline text")
def cmd_train_text(engine: SLMEngine, trainer: SLMTrainer, args: List[str]):
    if not args:
        print(f"{C.RED}  Usage: train-text \"<your text here>\"{C.RESET}")
        return

    text = " ".join(args)
    trainer.train_text(text, label="inline text")
    print(f"{C.CYAN}  Model: {engine.vocab_size():,} unique words{C.RESET}")


@command("generate", 'generate "<prompt>" [--length N] [--temp T]', "Generate text from prompt")
def cmd_generate(engine: SLMEngine, trainer: SLMTrainer, args: List[str]):
    if not engine.is_trained:
        print(f"{C.YELLOW}  No model loaded. Train or load a model first.{C.RESET}")
        return

    if not args:
        print(f"{C.RED}  Usage: generate \"<prompt>\" [--length N] [--temp T]{C.RESET}")
        return

    # Parse arguments
    max_length = 50
    temperature = 0.7
    prompt_parts = []

    i = 0
    while i < len(args):
        if args[i] == "--length" and i + 1 < len(args):
            try:
                max_length = int(args[i + 1])
            except ValueError:
                pass
            i += 2
        elif args[i] == "--temp" and i + 1 < len(args):
            try:
                temperature = float(args[i + 1])
            except ValueError:
                pass
            i += 2
        else:
            prompt_parts.append(args[i])
            i += 1

    prompt = " ".join(prompt_parts)

    print(f"\n{C.GRAY}  Prompt: {prompt}{C.RESET}")
    print(f"{C.GRAY}  Length: {max_length}, Temperature: {temperature}{C.RESET}\n")

    start = time.time()
    result = engine.generate(prompt, max_length, temperature)
    elapsed = time.time() - start

    print(f"  {C.GREEN}{C.BOLD}{result}{C.RESET}")
    print(f"\n{C.GRAY}  ({elapsed:.2f}s){C.RESET}")


@command("predict", 'predict "<context>"', "Show top next-word predictions")
def cmd_predict(engine: SLMEngine, trainer: SLMTrainer, args: List[str]):
    if not engine.is_trained:
        print(f"{C.YELLOW}  No model loaded. Train or load a model first.{C.RESET}")
        return

    if not args:
        print(f"{C.RED}  Usage: predict \"<context>\"{C.RESET}")
        return

    context = " ".join(args)
    predictions = engine.predict_next(context, 10)

    print(f"\n{C.CYAN}  Next word predictions for \"{context}\":{C.RESET}\n")

    if not predictions:
        print(f"  {C.YELLOW}No predictions available.{C.RESET}")
        return

    for i, (word, prob) in enumerate(predictions, 1):
        bar_len = int(prob * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        print(f"  {C.WHITE}{i:2d}. {word:20s} {C.BLUE}{bar} {C.CYAN}{prob:.4f}{C.RESET}")


@command("chat", "chat", "Interactive chat mode")
def cmd_chat(engine: SLMEngine, trainer: SLMTrainer, args: List[str]):
    if not engine.is_trained:
        print(f"{C.YELLOW}  No model loaded. Train or load a model first.{C.RESET}")
        return

    print(f"\n{C.CYAN}  ╔══════════════════════════════════════╗{C.RESET}")
    print(f"{C.CYAN}  ║     SLM Chat Mode                    ║{C.RESET}")
    print(f"{C.CYAN}  ║  Type your message, 'exit' to quit   ║{C.RESET}")
    print(f"{C.CYAN}  ╚══════════════════════════════════════╝{C.RESET}\n")

    while True:
        try:
            user_input = input(f"  {C.BLUE}You > {C.RESET}")
        except (EOFError, KeyboardInterrupt):
            print(f"\n{C.GRAY}  Chat ended.{C.RESET}")
            break

        if not user_input.strip():
            continue

        if user_input.strip().lower() in ("exit", "quit", "q"):
            print(f"{C.GRAY}  Chat ended.{C.RESET}")
            break

        result = engine.generate(user_input.strip(), max_length=60, temperature=0.7)
        print(f"  {C.GREEN}SLM > {result}{C.RESET}\n")


@command("save", "save <model_path>", "Save trained model to disk")
def cmd_save(engine: SLMEngine, trainer: SLMTrainer, args: List[str]):
    if not engine.is_trained:
        print(f"{C.YELLOW}  No model to save. Train first.{C.RESET}")
        return

    if not args:
        print(f"{C.RED}  Usage: save <model_path>{C.RESET}")
        return

    path = args[0]
    # Ensure directory exists
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    print(f"{C.CYAN}  Saving model to: {path}{C.RESET}")

    if engine.save_model(path):
        print(f"{C.GREEN}  ✓ Model saved successfully{C.RESET}")
    else:
        print(f"{C.RED}  ERROR: Failed to save model{C.RESET}")


@command("load", "load <model_path>", "Load a saved model")
def cmd_load(engine: SLMEngine, trainer: SLMTrainer, args: List[str]):
    if not args:
        print(f"{C.RED}  Usage: load <model_path>{C.RESET}")
        return

    path = args[0]

    if not os.path.exists(path):
        print(f"{C.RED}  ERROR: Model file not found: {path}{C.RESET}")
        return

    print(f"{C.CYAN}  Loading model from: {path}{C.RESET}")

    if engine.load_model(path):
        print(f"{C.GREEN}  ✓ Model loaded ({engine.vocab_size():,} words, {engine.total_tokens():,} tokens){C.RESET}")
    else:
        print(f"{C.RED}  ERROR: Failed to load model (invalid or corrupted file){C.RESET}")


@command("info", "info", "Show model statistics")
def cmd_info(engine: SLMEngine, trainer: SLMTrainer, args: List[str]):
    engine_type = get_engine_type()

    print(f"\n{C.CYAN}  ╔══════════════════════════════════════╗{C.RESET}")
    print(f"{C.CYAN}  ║        SLM Model Info                ║{C.RESET}")
    print(f"{C.CYAN}  ╚══════════════════════════════════════╝{C.RESET}")
    print(f"  {C.WHITE}Engine:{C.RESET}           {engine_type}")
    print(f"  {C.WHITE}Version:{C.RESET}          {__version__}")
    print(f"  {C.WHITE}Status:{C.RESET}           {'Trained' if engine.is_trained else 'Not trained'}")
    print(f"  {C.WHITE}Vocabulary:{C.RESET}       {engine.vocab_size():,} unique words")
    print(f"  {C.WHITE}Total Tokens:{C.RESET}     {engine.total_tokens():,}")
    print(f"  {C.WHITE}Unigrams:{C.RESET}         {engine.model.unigrams.size():,}")
    print(f"  {C.WHITE}Bigrams:{C.RESET}          {engine.model.bigrams.size():,}")
    print(f"  {C.WHITE}Trigrams:{C.RESET}         {engine.model.trigrams.size():,}")
    print()


@command("vocab", "vocab [--top N]", "Show top vocabulary words")
def cmd_vocab(engine: SLMEngine, trainer: SLMTrainer, args: List[str]):
    if not engine.is_trained:
        print(f"{C.YELLOW}  No model loaded. Train or load a model first.{C.RESET}")
        return

    n = 20
    if "--top" in args:
        idx = args.index("--top")
        if idx + 1 < len(args):
            try:
                n = int(args[idx + 1])
            except ValueError:
                pass

    top = engine.top_vocab(n)

    print(f"\n{C.CYAN}  Top {len(top)} words by frequency:{C.RESET}\n")

    if not top:
        print(f"  {C.YELLOW}No vocabulary data.{C.RESET}")
        return

    max_count = top[0][1] if top else 1

    for i, (word, count) in enumerate(top, 1):
        bar_len = int((count / max_count) * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {C.WHITE}{i:3d}. {word:20s} {C.BLUE}{bar} {C.CYAN}{count:,}{C.RESET}")

    print()


@command("reset", "reset", "Clear model from memory")
def cmd_reset(engine: SLMEngine, trainer: SLMTrainer, args: List[str]):
    engine.reset()
    print(f"{C.GREEN}  ✓ Model cleared{C.RESET}")


@command("help", "help", "Show all available commands")
def cmd_help(engine: SLMEngine, trainer: SLMTrainer, args: List[str]):
    print(f"\n{C.CYAN}  ╔══════════════════════════════════════════════════════════╗{C.RESET}")
    print(f"{C.CYAN}  ║                 SLM Commands                             ║{C.RESET}")
    print(f"{C.CYAN}  ╚══════════════════════════════════════════════════════════╝{C.RESET}\n")

    for name, info in sorted(COMMANDS.items()):
        print(f"  {C.GREEN}{info['usage']:45s}{C.RESET} {C.GRAY}{info['description']}{C.RESET}")

    print(f"\n  {C.GRAY}Type 'exit' or 'quit' to exit the SLM shell.{C.RESET}\n")


@command("version", "version", "Show SLM version")
def cmd_version(engine: SLMEngine, trainer: SLMTrainer, args: List[str]):
    print(f"\n  {C.CYAN}SLM — Singular Language Model{C.RESET}")
    print(f"  {C.WHITE}Version: {__version__}{C.RESET}")
    print(f"  {C.WHITE}Engine:  {get_engine_type()}{C.RESET}")
    print(f"  {C.GRAY}Offline frequency-based language model{C.RESET}\n")


def execute_command(
    command_name: str,
    engine: SLMEngine,
    trainer: SLMTrainer,
    args: List[str],
):
    """Execute a registered command by name."""
    if command_name in COMMANDS:
        COMMANDS[command_name]["func"](engine, trainer, args)
    else:
        print(f"{C.RED}  Unknown command: '{command_name}'{C.RESET}")
        print(f"{C.GRAY}  Type 'help' to see available commands.{C.RESET}")
