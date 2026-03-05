"""
SLM Config — Manages persistent configuration and state.

Stores: API keys, model paths, default settings.
File:   slm_config.json  (in project root)
"""

import os
import json
from typing import Any, Optional

# Location of config file (always next to slm_cli.py)
_CONFIG_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "slm_config.json"
)
_CONFIG_FILE = os.path.normpath(_CONFIG_FILE)

_DEFAULTS = {
    "version": "1.0.0",
    "gemini_api_key": "",
    "gemini_model": "gemini-1.5-flash",
    "default_model_path": "models/default.slm",
    "default_temp": 0.7,
    "default_length": 50,
    "auto_load": True,
    "corpus_dir": "corpus",
    "last_model": "",
}


def load() -> dict:
    """Load config from file, returning defaults for missing keys."""
    config = dict(_DEFAULTS)
    if os.path.exists(_CONFIG_FILE):
        try:
            with open(_CONFIG_FILE, "r", encoding="utf-8") as f:
                stored = json.load(f)
            config.update(stored)
        except (json.JSONDecodeError, IOError):
            pass
    return config


def save(config: dict) -> bool:
    """Save config to file."""
    try:
        with open(_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        return True
    except IOError:
        return False


def get(key: str, default: Any = None) -> Any:
    """Get a single config value."""
    return load().get(key, default)


def set_value(key: str, value: Any) -> bool:
    """Set a single config value and persist."""
    config = load()
    config[key] = value
    return save(config)


def get_api_key() -> Optional[str]:
    """Return Gemini API key if configured, else None."""
    key = get("gemini_api_key", "")
    return key if key else None


def reset() -> bool:
    """Reset config to defaults."""
    return save(dict(_DEFAULTS))


def path() -> str:
    """Return the config file path."""
    return _CONFIG_FILE
