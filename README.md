# ███████╗██╗     ███╗   ███╗
# ██╔════╝██║     ████╗ ████║
# ███████╗██║     ██╔████╔██║
# ╚════██║██║     ██║╚██╔╝██║
# ███████║███████╗██║ ╚═╝ ██║
# ╚══════╝╚══════╝╚═╝     ╚═╝

# SLM — Singular Language Model

> **The Free World's AI** — A fully offline, frequency-based language model.
>fae No internet. No cloud. No APIs. No GPU. Just local intelligence.

---

## What is SLM?

SLM is a **Singular Language Model** — a lightweight AI that runs 100% on your local machine with zero network dependencies. Unlike modern neural-network LLMs (ChatGPT, LLaMA, etc.), SLM uses **n-gram frequency analysis** — the simplest and most transparent form of language intelligence.

**You feed it text → It learns word patterns → It generates new text.**

That's it. No magic. No black box. Pure statistics.

---

## How It Works and who we can use it 

### The Core Idea: N-Gram Frequency Analysis

SLM learns by counting how often word sequences appear in training text.

```
Training text: "the world is vast and the world is beautiful"

Unigrams (single words):
  the → 2    world → 2    is → 2    vast → 1    and → 1    beautiful → 1

Bigrams (2-word pairs):
  the world    → 2
  world is     → 2
  is vast      → 1
  vast and     → 1
  and the      → 1
  is beautiful → 1

Trigrams (3-word sequences):
  the world is     → 2
  world is vast    → 1
  is vast and      → 1
  vast and the     → 1
  and the world    → 1
  world is beautiful → 1
```

### How Generation Works

When you give SLM a prompt like `"the world is"`, it:

1. **Looks up trigrams** — finds `"the world is"` → next word could be `vast` (50%) or `beautiful` (50%)
2. **If no trigram match** — falls back to bigrams: looks at `"world is"` → same result
3. **If no bigram match** — falls back to unigrams: picks from most common words
4. **Temperature sampling** — controls randomness (low = predictable, high = creative)
5. **Repeats** — chains predictions to build full sentences

### The Pipeline

```
┌──────────┐    ┌───────────┐    ┌──────────────────┐    ┌───────────┐
│  Raw Text │───>│ Tokenizer │───>│  N-Gram Model    │───>│ Generator │
│  (input)  │    │ lowercase │    │  count unigrams  │    │ weighted  │
│           │    │ strip punct│    │  count bigrams   │    │ random    │
│           │    │ split words│    │  count trigrams  │    │ sampling  │
└──────────┘    └───────────┘    └──────────────────┘    └───────────┘
                                         │
                                    ┌────┴────┐
                                    │ .slm    │
                                    │ file    │
                                    │ (save/  │
                                    │  load)  │
                                    └─────────┘
```

---

## Quick Start

```bash
# Just run it — no installation needed, only Python 3
python slm_cli.py
```

This opens the **interactive SLM shell**. From there:

```
slm > train corpus/seed.txt          # Train on the bundled text
slm > generate "the world is"        # Generate text
slm > chat                           # Chat with SLM
slm > exit                           # Quit
```

Or run **single commands** directly:

```bash
python slm_cli.py train corpus/seed.txt
python slm_cli.py generate "the world is"
python slm_cli.py help
```

---

## All Commands

### `train <file|directory>`
Train the model on a text file or all `.txt` files in a directory.

```
slm > train corpus/seed.txt
  Training on: seed.txt (7,181 bytes)
  ✓ Processed 1,061 tokens in 0.00s (1,027,275 tokens/sec)
  Model: 565 unique words, 1,061 total tokens

slm > train my_books/
  Found 5 text file(s) in my_books/
  [1/5] book1.txt (12,340 tokens)
  [2/5] book2.txt (8,921 tokens)
  ...
```

---

### `train-text "<text>"`
Train on inline text without needing a file.

```
slm > train-text "the quick brown fox jumps over the lazy dog"
  Training on inline text (44 chars)
  ✓ Processed 9 tokens in 0.00s
```

---

### `generate "<prompt>" [--length N] [--temp T]`
Generate text continuation from a prompt.

- `--length N` — max words to generate (default: 50)
- `--temp T` — temperature 0.1-2.0 (default: 0.7, lower = more predictable)

```
slm > generate "the world is"
  Prompt: the world is
  Length: 50, Temperature: 0.7

  the world is a vast and interconnected place where people from different
  cultures and backgrounds come together to share ideas and build communities

slm > generate "science has" --length 30 --temp 0.3
  science has helped us understand the fundamental laws that govern our
  universe from the smallest atoms to the largest galaxies
```

---

### `predict "<context>"`
Show the top next-word predictions with probability bars.

```
slm > predict "the"
  Next word predictions for "the":

   1. world               █████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0.1200
   2. most                ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0.0800
   3. power               ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0.0600
   4. way                 █████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0.0500
   5. full                ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0.0400
```

---

### `chat`
Interactive conversation mode. Type messages, get SLM responses.

```
slm > chat
  ╔══════════════════════════════════════╗
  ║     SLM Chat Mode                    ║
  ║  Type your message, 'exit' to quit   ║
  ╚══════════════════════════════════════╝

  You > tell me about the world
  SLM > the world is a vast and interconnected place where people from 
        different cultures come together to share ideas

  You > what about science
  SLM > science has helped us understand the fundamental laws that govern
        our universe from the smallest atoms to the largest galaxies

  You > exit
  Chat ended.
```

---

### `save <model_path>`
Save the trained model to a file for later use.

```
slm > save models/my_model.slm
  Saving model to: models/my_model.slm
  ✓ Model saved successfully
```

---

### `load <model_path>`
Load a previously saved model.

```
slm > load models/my_model.slm
  Loading model from: models/my_model.slm
  ✓ Model loaded (565 words, 1,061 tokens)
```

---

### `info`
Show current model statistics.

```
slm > info
  ╔══════════════════════════════════════╗
  ║        SLM Model Info                ║
  ╚══════════════════════════════════════╝
  Engine:           Pure Python (fallback)
  Version:          1.0.0
  Status:           Trained
  Vocabulary:       565 unique words
  Total Tokens:     1,061
  Unigrams:         565
  Bigrams:          948
  Trigrams:         1,003
```

---

### `vocab [--top N]`
Show the most frequent words in the model.

```
slm > vocab --top 10
  Top 10 words by frequency:

    1. the                  ██████████████████████████████ 72
    2. and                  ████████████████████████░░░░░░ 56
    3. of                   ██████████████████░░░░░░░░░░░░ 41
    4. to                   █████████████████░░░░░░░░░░░░░ 38
    5. a                    ██████████████░░░░░░░░░░░░░░░░ 30
```

---

### `reset`
Clear all model data from memory.

```
slm > reset
  ✓ Model cleared
```

---

### `help`
Show all available commands.

---

### `version`
Show SLM version and engine info.

```
slm > version
  SLM — Singular Language Model
  Version: 1.0.0
  Engine:  Pure Python (fallback)
  Offline frequency-based language model
```

---

## Typical Workflow

```bash
# Step 1: Launch SLM
python slm_cli.py

# Step 2: Train on text (more text = better results)
slm > train corpus/seed.txt
slm > train-text "any additional text you want to teach it"

# Step 3: Use it
slm > generate "once upon a time"
slm > chat
slm > predict "the future of"

# Step 4: Save your trained model
slm > save models/my_brain.slm

# Step 5: Next time, just load it
slm > load models/my_brain.slm
slm > generate "hello world"
```

### Training Tips
- **More text = better results** — Feed it books, articles, your own writing
- **Domain-specific training** — Train on science text for science generation, poetry for poetry, etc.
- **Multiple files** — Put `.txt` files in a folder and use `train <folder>/`
- **Low temperature (0.1-0.3)** — More predictable, repetitive output
- **High temperature (0.8-1.5)** — More creative, sometimes nonsensical output

---

## Project Structure

```
SLM/
├── slm_cli.py                  # ← RUN THIS — Main entry point
├── README.md                   # This file
├── test_slm.py                 # Integration tests
│
├── python/slm/                 # Python package
│   ├── __init__.py             # Package init
│   ├── engine.py               # Pure Python n-gram engine (Tokenizer, NGramModel, Generator)
│   ├── cli.py                  # CLI interface (REPL + single-command mode)
│   ├── commands.py             # All 12 command handlers
│   ├── trainer.py              # Training pipeline with progress output
│   └── bridge.py               # C++ engine bridge (optional)
│
├── core/                       # C++ engine (optional, for speed)
│   ├── include/slm/            # Headers
│   │   ├── tokenizer.h
│   │   ├── frequency_store.h
│   │   ├── ngram_model.h
│   │   ├── generator.h
│   │   └── slm_api.h
│   └── src/                    # Source files
│       ├── tokenizer.cpp
│       ├── frequency_store.cpp
│       ├── ngram_model.cpp
│       ├── generator.cpp
│       └── slm_api.cpp
│
├── corpus/                     # Training data
│   └── seed.txt                # Built-in starter corpus (~1000 words)
│
├── models/                     # Saved model files
│
├── CMakeLists.txt              # C++ build config
├── build.bat                   # Windows C++ build script
└── build.sh                    # Linux/Mac C++ build script
```

---

## Two Engines

| Feature | Python Engine | C++ Engine |
|---------|--------------|------------|
| **Works out-of-box** | ✅ Yes | ❌ Needs compiler |
| **Speed** | Good for small-medium text | 10-100x faster for large corpora |
| **Dependencies** | Python 3 only | CMake + MSVC/GCC/Clang |
| **Build command** | None needed | `build.bat` (Windows) / `build.sh` (Linux) |

The Python engine is the default and **requires nothing except Python 3**. The C++ engine is auto-detected if built.

### Building the C++ Engine (Optional)

```bash
# Windows (needs Visual Studio / MSVC + CMake)
build.bat

# Linux/Mac (needs g++ or clang++ + CMake)
chmod +x build.sh && ./build.sh
```

---

## Requirements

- **Python 3.6+** — That's it. No pip packages. No virtual environments. Zero dependencies.

---

## Version

**SLM v1.0.0** — Singular Language Model  

Co-authored-by: Chandan Sharma <mrchandansharma25@gmail.com>
Co-authored-by: Chandan Sharma <mrchandansharma26@gmail.com>


Offline · Frequency-Based · Free · Open
