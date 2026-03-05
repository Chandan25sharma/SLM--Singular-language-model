"""
SLM Config — Manages persistent configuration and state.

Priority order for reading values:
  1. .env file (GOOGLE_API_KEY, GEMINI_MODEL, ...)
  2. slm_config.json
  3. Built-in defaults

File: slm_config.json (in project root, git-ignored)
      .env            (in project root, git-ignored)
"""

import os
import json
from typing import Any, Optional

# Root of the project (two levels up from this file)
_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
_CONFIG_FILE = os.path.join(_ROOT, "slm_config.json")
_ENV_FILE    = os.path.join(_ROOT, ".env")

_DEFAULTS = {
    "version":            "1.0.0",
    "gemini_model":       "gemini-2.0-flash",
    "default_model_path": "models/default.slm",
    "default_temp":       0.7,
    "default_length":     50,
    "auto_load":          True,
    "corpus_dir":         "corpus",
    "last_model":         "",
}

# Keys that must ONLY live in .env — never written to JSON
_ENV_ONLY_KEYS = {"gemini_api_key"}

# Mapping from .env variable names → config keys
_ENV_MAP = {
    "GOOGLE_API_KEY":  "gemini_api_key",
    "GEMINI_API_KEY":  "gemini_api_key",
    "GEMINI_MODEL":    "gemini_model",
}


def _load_env() -> dict:
    """Read key=value pairs from .env file (no external deps)."""
    result = {}
    if not os.path.exists(_ENV_FILE):
        return result
    try:
        with open(_ENV_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                env_key = _ENV_MAP.get(key)
                if env_key:
                    result[env_key] = val
    except IOError:
        pass
    return result


def load() -> dict:
    """Load config: defaults → slm_config.json → .env (highest priority)."""
    config = dict(_DEFAULTS)

    # Layer 1: slm_config.json
    if os.path.exists(_CONFIG_FILE):
        try:
            with open(_CONFIG_FILE, "r", encoding="utf-8") as f:
                stored = json.load(f)
            config.update(stored)
        except (json.JSONDecodeError, IOError):
            pass

    # Layer 2: .env overrides (highest priority)
    config.update(_load_env())
    return config


def save(config: dict) -> bool:
    """Save config to slm_config.json (excludes .env values)."""
    try:
        with open(_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        return True
    except IOError:
        return False


def get(key: str, default: Any = None) -> Any:
    """Get a single config value (respects .env priority)."""
    return load().get(key, default)


def set_value(key: str, value: Any) -> bool:
    """Persist a value to slm_config.json. API keys are blocked — use .env instead."""
    if key in _ENV_ONLY_KEYS:
        raise ValueError(
            f"'{key}' is secret and must be stored in .env only, not in slm_config.json.\n"
            f"Edit your .env file and set: GOOGLE_API_KEY=your_key_here"
        )
    config = load()
    config[key] = value
    return save(config)


def get_api_key() -> Optional[str]:
    """Return Gemini API key — reads .env first, then slm_config.json."""
    key = get("gemini_api_key", "")
    return key.strip() if key and key.strip() else None


def reset() -> bool:
    """Reset slm_config.json to defaults (does not touch .env)."""
    return save(dict(_DEFAULTS))


def path() -> str:
    return _CONFIG_FILE


def env_path() -> str:
    return _ENV_FILE
