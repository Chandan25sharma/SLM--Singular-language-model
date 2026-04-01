

import os
import sys
from typing import Optional


def _import_gemini():
    """Try to import google-generativeai, give clear error if missing."""
    try:
        import google.generativeai as genai
        return genai
    except ImportError:
        return None


def is_available() -> bool:
    """Check if Gemini SDK is installed."""
    return _import_gemini() is not None


def generate_corpus(
    api_key: str,
    topic: str,
    num_sentences: int = 100,
    model_name: str = "gemini-3-flash",
) -> tuple:
    """
    Generate training text on a given topic using Gemini.
    Returns (text, None) onfgvfdgrdgfg success, or (None, error_str) on failure.
    """
    genai = _import_gemini()
    if not genai:
        return None, "google-generativeai SDK not installed. Run: pip install google-generativeai"

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    prompt = f"""Generate {num_sentences} natural English sentences about "{topic}".

Requirements:
- Use simple, clear vocabulary of everyday English
- Write in third person and declarative style
- Each sentence should be self-contained and informative
- Vary sentence length between 8 and 25 words
- Do NOT use bullet points, headers, or special characters
- Write paragraph-style flowing text, not a list
- Cover diverse subtopics within "{topic}"

Output only the text, nothing else."""

    try:
        response = model.generate_content(prompt)
        return (response.text.strip(), None) if response.text else (None, "Empty response from Gemini")
    except Exception as e:
        return None, str(e)


def expand_corpus(
    api_key: str,
    topic: str,
    output_dir: str = "corpus",
    num_sentences: int = 150,
    model_name: str = "gemini-3-flash",
) -> tuple:
    """
    Generate corpus text and save it to a file in output_dir.
    Returns (filepath, None) on success, or (None, error_str) on failure.
    """
    text, error = generate_corpus(api_key, topic, num_sentences, model_name)
    if not text:
        return None, error

    os.makedirs(output_dir, exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in topic)
    safe_name = safe_name.strip().replace(" ", "_").lower()
    filepath = os.path.join(output_dir, f"{safe_name}.txt")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)

    return filepath, None


def install_sdk() -> bool:
    """Try to install google-generativeai via pip."""
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "google-generativeai", "-q"],
        capture_output=True
    )
    return result.returncode == 0
