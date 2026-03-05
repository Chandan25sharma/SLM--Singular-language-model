"""
SLM Commands - Command registry and handlers for the SLM CLI.
"""

import os
import sys
import time
from typing import List
from .engine import SLMEngine
from .trainer import SLMTrainer
from .bridge import get_engine_type
from . import __version__


# ─── ANSI Colors ─────────────────────────────────────────────
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


def _ok(msg):  print(f"  {C.GREEN}✓ {msg}{C.RESET}")
def _err(msg): print(f"  {C.RED}✗ ERROR: {msg}{C.RESET}")
def _tip(msg): print(f"  {C.YELLOW}→ {msg}{C.RESET}")
def _sep():    print(f"  {C.GRAY}{'─' * 50}{C.RESET}")


# ─── Command Registry ─────────────────────────────────────────
COMMANDS = {}


def command(name: str, usage: str, description: str, aliases=None):
    """Decorator to register a command."""
    def decorator(func):
        COMMANDS[name] = {
            "func": func,
            "usage": usage,
            "description": description,
        }
        for alias in (aliases or []):
            COMMANDS[alias] = COMMANDS[name]
        return func
    return decorator


# ─── Helpers ─────────────────────────────────────────────────

def _need_model(engine: SLMEngine) -> bool:
    """Check if model is trained; print a helpful message if not."""
    if not engine.is_trained:
        print(f"\n  {C.YELLOW}⚠ No model loaded yet.{C.RESET}")
        print(f"  {C.GRAY}Train one first:{C.RESET}")
        print(f"  {C.CYAN}  → train corpus/seed.txt{C.RESET}")
        print(f"  {C.CYAN}  → quickstart{C.RESET}  (auto-trains + generates)\n")
        return False
    return True


def _parse_flags(args: List[str], *flag_defs):
    """
    Parse named flags from argument list.
    flag_defs: tuples of (flag_name, default_value, converter)
    Returns (remaining_args, {flag: value})
    """
    values = {f[0]: f[1] for f in flag_defs}
    remaining = []
    i = 0
    while i < len(args):
        matched = False
        for flag, default, conv in flag_defs:
            if args[i] == flag and i + 1 < len(args):
                try:
                    values[flag] = conv(args[i + 1])
                except (ValueError, TypeError):
                    _err(f"Invalid value for {flag}: '{args[i+1]}'")
                i += 2
                matched = True
                break
        if not matched:
            remaining.append(args[i])
            i += 1
    return remaining, values


# ─── Commands ─────────────────────────────────────────────────

@command("quickstart", "quickstart", "Auto-train on seed corpus and generate sample text")
def cmd_quickstart(engine: SLMEngine, trainer: SLMTrainer, args: List[str]):
    """One-command first run — trains and shows results."""
    seed_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "corpus", "seed.txt"
    )
    seed_path = os.path.normpath(seed_path)

    if not os.path.exists(seed_path):
        _err(f"Seed corpus not found: {seed_path}")
        return

    print(f"\n  {C.CYAN}{'━' * 50}{C.RESET}")
    print(f"  {C.CYAN}  SLM Quickstart{C.RESET}")
    print(f"  {C.CYAN}{'━' * 50}{C.RESET}\n")

    trainer.train_file(seed_path)

    print()
    _sep()
    examples = [
        ("the world is", 25, 0.8),
        ("science has", 20, 0.7),
        ("language is", 20, 0.7),
    ]
    for prompt, ln, tmp in examples:
        result = engine.generate(prompt, ln, tmp)
        print(f"  {C.GRAY}Prompt:{C.RESET} {C.WHITE}{prompt}{C.RESET}")
        print(f"  {C.GREEN}{result}{C.RESET}\n")

    _sep()
    print(f"\n  {C.CYAN}Model ready: {engine.vocab_size():,} words  |  {engine.total_tokens():,} tokens{C.RESET}")
    print(f"  {C.GRAY}Try: generate \"<your prompt>\"  │  chat  │  save models/my.slm{C.RESET}\n")


@command("train", "train <file|dir>", "Train model on text file(s)", aliases=["t"])
def cmd_train(engine: SLMEngine, trainer: SLMTrainer, args: List[str]):
    if not args:
        _err("Path required")
        _tip('train corpus/seed.txt')
        _tip('train corpus/')
        return

    target = args[0]

    if os.path.isfile(target):
        trainer.train_file(target)
    elif os.path.isdir(target):
        trainer.train_directory(target)
    else:
        _err(f"Not found: '{target}'")
        _tip("Check spelling or use a full path")
        return

    print()
    _ok(f"Model updated: {C.WHITE}{engine.vocab_size():,} unique words  |  {engine.total_tokens():,} tokens")


@command("train-text", 'train-text "<text>"', "Train on inline text", aliases=["tt"])
def cmd_train_text(engine: SLMEngine, trainer: SLMTrainer, args: List[str]):
    if not args:
        _err("Text required")
        _tip('train-text "machine learning is the future"')
        return

    text = " ".join(args)
    trainer.train_text(text, label="inline text")
    _ok(f"Learned from inline text  |  {engine.vocab_size():,} unique words")


@command("generate", 'generate "<prompt>" [--length N] [--temp T]', "Generate text from prompt", aliases=["gen", "g"])
def cmd_generate(engine: SLMEngine, trainer: SLMTrainer, args: List[str]):
    if not _need_model(engine):
        return

    if not args:
        _err("Prompt required")
        _tip('generate "the future of science"')
        _tip('generate "once upon a" --length 80 --temp 1.0')
        return

    remaining, flags = _parse_flags(
        args,
        ("--length", 50, int),
        ("--temp",   0.7, float),
    )
    prompt = " ".join(remaining)
    max_length = flags["--length"]
    temperature = flags["--temp"]

    print(f"\n  {C.GRAY}▶ Prompt: {C.WHITE}{prompt}")
    print(f"  {C.GRAY}  Length: {max_length}  |  Temperature: {temperature}{C.RESET}\n")

    start = time.time()
    result = engine.generate(prompt, max_length, temperature)
    elapsed = time.time() - start

    # Pretty-print the result in a box
    words = result.split()
    line, lines = [], []
    for w in words:
        line.append(w)
        if len(" ".join(line)) > 70:
            lines.append(" ".join(line))
            line = []
    if line:
        lines.append(" ".join(line))

    print(f"  {C.CYAN}╭{'─' * 66}╮{C.RESET}")
    for ln in lines:
        print(f"  {C.CYAN}│{C.RESET} {C.GREEN}{C.BOLD}{ln:<65}{C.RESET} {C.CYAN}│{C.RESET}")
    print(f"  {C.CYAN}╰{'─' * 66}╯{C.RESET}")
    print(f"\n  {C.GRAY}({elapsed:.2f}s  |  {len(result.split())} words){C.RESET}\n")


@command("predict", 'predict "<context>"', "Show top next-word predictions", aliases=["p"])
def cmd_predict(engine: SLMEngine, trainer: SLMTrainer, args: List[str]):
    if not _need_model(engine):
        return

    if not args:
        _err("Context required")
        _tip('predict "the world"')
        return

    context = " ".join(args)
    predictions = engine.predict_next(context, 10)

    print(f"\n  {C.CYAN}Next words after → {C.WHITE}{C.BOLD}\"{context}\"{C.RESET}\n")

    if not predictions:
        print(f"  {C.YELLOW}No predictions (try training on more text){C.RESET}")
        return

    max_prob = predictions[0][1] if predictions else 1
    for i, (word, prob) in enumerate(predictions, 1):
        bar_len = int((prob / max_prob) * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        pct = f"{prob * 100:.1f}%"
        rank_col = C.GREEN if i == 1 else (C.YELLOW if i <= 3 else C.WHITE)
        print(f"  {rank_col}{i:2d}. {word:20s}{C.RESET}  {C.BLUE}{bar}{C.RESET}  {C.CYAN}{pct:>6}{C.RESET}")
    print()


@command("chat", "chat", "Interactive chat mode — type prompts, get continuations")
def cmd_chat(engine: SLMEngine, trainer: SLMTrainer, args: List[str]):
    if not _need_model(engine):
        return

    remaining, flags = _parse_flags(args, ("--temp", 0.7, float), ("--length", 60, int))
    temperature = flags["--temp"]
    max_length  = flags["--length"]

    print(f"\n  {C.CYAN}{'━' * 50}{C.RESET}")
    print(f"  {C.CYAN}  Chat Mode  {C.GRAY}(Ctrl+C or 'exit' to quit){C.RESET}")
    print(f"  {C.GRAY}  Temperature: {temperature}  |  Max length: {max_length}{C.RESET}")
    print(f"  {C.CYAN}{'━' * 50}{C.RESET}\n")

    history = []
    while True:
        try:
            user_input = input(f"  {C.BLUE}You › {C.RESET}")
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {C.GRAY}Chat ended.{C.RESET}\n")
            break

        user_input = user_input.strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print(f"  {C.GRAY}Chat ended.{C.RESET}\n")
            break

        # Use rolling context for better continuation
        context = " ".join(history[-4:] + [user_input]) if history else user_input
        result = engine.generate(context, max_length=max_length, temperature=temperature)
        history.append(user_input)

        print(f"  {C.GREEN}SLM › {result}{C.RESET}\n")


@command("save", "save <model_path>", "Save trained model to disk", aliases=["s"])
def cmd_save(engine: SLMEngine, trainer: SLMTrainer, args: List[str]):
    if not _need_model(engine):
        return

    path = args[0] if args else "models/model.slm"
    if not args:
        _tip(f"No path given, saving to: {path}")

    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    print(f"\n  {C.CYAN}Saving → {path}{C.RESET}")

    if engine.save_model(path):
        size_kb = os.path.getsize(path) // 1024 if os.path.exists(path) else 0
        _ok(f"Saved  ({size_kb} KB)")
    else:
        _err("Save failed — check disk space and path")


@command("load", "load <model_path>", "Load a saved model from disk", aliases=["l"])
def cmd_load(engine: SLMEngine, trainer: SLMTrainer, args: List[str]):
    if not args:
        # Auto-detect available models
        models_dir = "models"
        if os.path.isdir(models_dir):
            slm_files = [f for f in os.listdir(models_dir) if f.endswith(".slm")]
            if slm_files:
                _tip(f"Available models in models/:")
                for f in slm_files:
                    print(f"   {C.WHITE}→ load models/{f}{C.RESET}")
            else:
                _tip("No saved models yet. Train and save first.")
        else:
            _err("Path required: load <model_path>")
        return

    path = args[0]

    if not os.path.exists(path):
        _err(f"File not found: {path}")
        # Auto-suggest similar files
        parent = os.path.dirname(path) or "."
        if os.path.isdir(parent):
            matches = [f for f in os.listdir(parent) if f.endswith(".slm")]
            if matches:
                _tip(f"Did you mean one of these?")
                for f in matches:
                    print(f"   {C.WHITE}→ load {os.path.join(parent, f)}{C.RESET}")
        return

    print(f"\n  {C.CYAN}Loading ← {path}{C.RESET}")

    if engine.load_model(path):
        _ok(f"Loaded  {C.WHITE}{engine.vocab_size():,} words  |  {engine.total_tokens():,} tokens")
    else:
        _err("Load failed — file may be corrupted or incompatible")


@command("info", "info", "Show model statistics and engine info")
def cmd_info(engine: SLMEngine, trainer: SLMTrainer, args: List[str]):
    engine_type = get_engine_type()
    status_col = C.GREEN if engine.is_trained else C.YELLOW
    status_txt = "Trained ✓" if engine.is_trained else "Not trained"

    print(f"\n  {C.CYAN}{'━' * 40}{C.RESET}")
    print(f"  {C.CYAN}  SLM Model Info{C.RESET}")
    print(f"  {C.CYAN}{'━' * 40}{C.RESET}")
    print(f"  {C.GRAY}Version    {C.RESET}{C.WHITE}{__version__}{C.RESET}")
    print(f"  {C.GRAY}Engine     {C.RESET}{C.WHITE}{engine_type}{C.RESET}")
    print(f"  {C.GRAY}Status     {C.RESET}{status_col}{status_txt}{C.RESET}")

    if engine.is_trained:
        print()
        print(f"  {C.GRAY}Vocabulary {C.RESET}{C.WHITE}{engine.vocab_size():,} unique words{C.RESET}")
        print(f"  {C.GRAY}Tokens     {C.RESET}{C.WHITE}{engine.total_tokens():,} total{C.RESET}")
        # Show n-gram stats only for Python backend
        if hasattr(engine, "_backend") and hasattr(engine._backend, "model"):
            m = engine._backend.model
            print(f"  {C.GRAY}Bigrams    {C.RESET}{C.WHITE}{m.bigrams.size():,}{C.RESET}")
            print(f"  {C.GRAY}Trigrams   {C.RESET}{C.WHITE}{m.trigrams.size():,}{C.RESET}")
    print()


@command("vocab", "vocab [--top N]", "Show top vocabulary words by frequency", aliases=["v"])
def cmd_vocab(engine: SLMEngine, trainer: SLMTrainer, args: List[str]):
    if not _need_model(engine):
        return

    _, flags = _parse_flags(args, ("--top", 20, int))
    n = flags["--top"]
    top = engine.top_vocab(n)

    if not top:
        print(f"  {C.YELLOW}No vocabulary data.{C.RESET}")
        return

    print(f"\n  {C.CYAN}Top {len(top)} words by frequency:{C.RESET}\n")

    max_count = top[0][1] if top else 1
    for i, (word, count) in enumerate(top, 1):
        bar_len = int((count / max_count) * 28)
        bar = "█" * bar_len + "░" * (28 - bar_len)
        rank_col = C.GREEN if i <= 3 else (C.YELLOW if i <= 10 else C.WHITE)
        print(f"  {rank_col}{i:3d}. {word:18s}{C.RESET}  {C.BLUE}{bar}{C.RESET}  {C.CYAN}{count:,}{C.RESET}")
    print()


@command("reset", "reset", "Clear model from memory")
def cmd_reset(engine: SLMEngine, trainer: SLMTrainer, args: List[str]):
    if not engine.is_trained:
        _tip("Model is already empty")
        return

    size = engine.vocab_size()
    engine.reset()
    _ok(f"Model cleared ({size:,} words removed)")


@command("help", "help [command]", "Show commands or help for a specific command", aliases=["h", "?"])
def cmd_help(engine: SLMEngine, trainer: SLMTrainer, args: List[str]):
    if args and args[0] in COMMANDS:
        info = COMMANDS[args[0]]
        print(f"\n  {C.CYAN}{info['usage']}{C.RESET}")
        print(f"  {C.GRAY}{info['description']}{C.RESET}\n")
        return

    print(f"\n  {C.CYAN}{'━' * 55}{C.RESET}")
    print(f"  {C.CYAN}  SLM Commands{C.RESET}")
    print(f"  {C.CYAN}{'━' * 55}{C.RESET}\n")

    # Group commands
    groups = {
        "Training":   ["quickstart", "train", "train-text"],
        "Generation": ["generate", "chat", "predict"],
        "Model":      ["save", "load", "reset", "info", "vocab"],
        "Other":      ["help", "version"],
    }

    printed = set()
    for group, names in groups.items():
        print(f"  {C.YELLOW}{group}{C.RESET}")
        for name in names:
            if name in COMMANDS and name not in printed:
                info = COMMANDS[name]
                print(f"  {C.WHITE}  {info['usage']:<40}{C.RESET}{C.GRAY}{info['description']}{C.RESET}")
                printed.add(name)
        print()

    print(f"  {C.GRAY}Aliases: gen/g, t, tt, p, s, l, v, h, ?{C.RESET}")
    print(f"  {C.GRAY}Type 'exit' or Ctrl+C to quit.{C.RESET}\n")


@command("version", "version", "Show SLM version and engine info")
def cmd_version(engine: SLMEngine, trainer: SLMTrainer, args: List[str]):
    engine_col = C.GREEN if "C++" in get_engine_type() else C.YELLOW
    print(f"\n  {C.CYAN}SLM — Singular Language Model{C.RESET}")
    print(f"  {C.WHITE}Version:{C.RESET} {__version__}")
    print(f"  {C.WHITE}Engine: {C.RESET} {engine_col}{get_engine_type()}{C.RESET}")
    print(f"  {C.GRAY}Offline · Frequency-based · No internet{C.RESET}\n")


def execute_command(
    command_name: str,
    engine: SLMEngine,
    trainer: SLMTrainer,
    args: List[str],
):
    """Execute a registered command by name."""
    if command_name in COMMANDS:
        try:
            COMMANDS[command_name]["func"](engine, trainer, args)
        except KeyboardInterrupt:
            print(f"\n  {C.YELLOW}Interrupted.{C.RESET}")
    else:
        print(f"\n  {C.RED}Unknown command: '{command_name}'{C.RESET}")
        # Fuzzy suggestion
        candidates = [k for k in COMMANDS if k.startswith(command_name[:2])]
        if candidates:
            print(f"  {C.GRAY}Did you mean: {', '.join(candidates[:3])}?{C.RESET}")
        print(f"  {C.GRAY}Type 'help' to see all commands.{C.RESET}\n")
