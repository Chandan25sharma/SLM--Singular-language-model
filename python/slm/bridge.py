"""
SLM Bridge - Loads C++ engine via ctypes, falls back to pure Python.
"""

import os
import sys
import ctypes
from typing import Optional

# Try to load C++ engine
_lib = None
_lib_path = None


def _find_library() -> Optional[str]:
    """Find the compiled SLM shared library."""
    base = os.path.dirname(os.path.abspath(__file__))
    
    if sys.platform == "win32":
        candidates = [
            os.path.join(base, "slm.dll"),
            os.path.join(base, "..", "..", "build", "Release", "slm.dll"),
            os.path.join(base, "..", "..", "build", "Debug", "slm.dll"),
        ]
    else:
        candidates = [
            os.path.join(base, "libslm.so"),
            os.path.join(base, "..", "..", "build", "libslm.so"),
            os.path.join(base, "libslm.dylib"),
        ]
    
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _load_library():
    """Attempt to load the C++ shared library."""
    global _lib, _lib_path
    
    path = _find_library()
    if not path:
        return False
    
    try:
        _lib = ctypes.CDLL(path)
        _lib_path = path
        
        # Set up function signatures
        _lib.slm_create.restype = ctypes.c_void_p
        _lib.slm_create.argtypes = []
        
        _lib.slm_destroy.restype = None
        _lib.slm_destroy.argtypes = [ctypes.c_void_p]
        
        _lib.slm_train_file.restype = ctypes.c_int
        _lib.slm_train_file.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        
        _lib.slm_train_text.restype = ctypes.c_int
        _lib.slm_train_text.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        
        _lib.slm_save_model.restype = ctypes.c_int
        _lib.slm_save_model.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        
        _lib.slm_load_model.restype = ctypes.c_int
        _lib.slm_load_model.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        
        _lib.slm_generate.restype = ctypes.c_void_p
        _lib.slm_generate.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_double]
        
        _lib.slm_predict_next.restype = ctypes.c_void_p
        _lib.slm_predict_next.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        
        _lib.slm_get_vocab_size.restype = ctypes.c_int
        _lib.slm_get_vocab_size.argtypes = [ctypes.c_void_p]
        
        _lib.slm_get_total_tokens.restype = ctypes.c_int
        _lib.slm_get_total_tokens.argtypes = [ctypes.c_void_p]
        
        _lib.slm_get_top_vocab.restype = ctypes.c_void_p
        _lib.slm_get_top_vocab.argtypes = [ctypes.c_void_p, ctypes.c_int]
        
        _lib.slm_free_string.restype = None
        _lib.slm_free_string.argtypes = [ctypes.c_void_p]
        
        _lib.slm_reset.restype = None
        _lib.slm_reset.argtypes = [ctypes.c_void_p]
        
        return True
    except Exception:
        _lib = None
        return False


def is_native_available() -> bool:
    """Check if the C++ engine is available."""
    if _lib is not None:
        return True
    return _load_library()


def get_engine_type() -> str:
    """Return which engine is being used."""
    if is_native_available():
        return f"C++ Native ({_lib_path})"
    return "Pure Python (fallback)"


class NativeEngine:
    """Python wrapper for the C++ SLM engine."""

    def __init__(self):
        if not is_native_available():
            raise RuntimeError("C++ engine library not found.")
        self._ctx = _lib.slm_create()

    def __del__(self):
        if hasattr(self, "_ctx") and self._ctx:
            _lib.slm_destroy(self._ctx)

    def train_file(self, filepath: str) -> int:
        res = _lib.slm_train_file(self._ctx, filepath.encode("utf-8"))
        if res != 0:
            return 0
        return self.total_tokens()  # Approximate tokens from total tokens (C++ API behavior)

    def train_text(self, text: str) -> int:
        res = _lib.slm_train_text(self._ctx, text.encode("utf-8"))
        if res != 0:
            return 0
        return self.total_tokens()

    def save_model(self, path: str) -> bool:
        return _lib.slm_save_model(self._ctx, path.encode("utf-8")) == 0

    def load_model(self, path: str) -> bool:
        return _lib.slm_load_model(self._ctx, path.encode("utf-8")) == 0

    def generate(self, prompt: str, max_length: int = 50, temperature: float = 0.7) -> str:
        res_ptr = _lib.slm_generate(self._ctx, prompt.encode("utf-8"), max_length, temperature)
        if not res_ptr:
            return ""
        result = ctypes.string_at(res_ptr).decode("utf-8")
        _lib.slm_free_string(res_ptr)
        return result

    def predict_next(self, context: str, num_candidates: int = 5) -> list:
        import json
        res_ptr = _lib.slm_predict_next(self._ctx, context.encode("utf-8"), num_candidates)
        if not res_ptr:
            return []
        json_str = ctypes.string_at(res_ptr).decode("utf-8")
        _lib.slm_free_string(res_ptr)
        try:
            preds = json.loads(json_str)
            return [(p["word"], p["prob"]) for p in preds]
        except Exception:
            return []

    def vocab_size(self) -> int:
        return _lib.slm_get_vocab_size(self._ctx)

    def total_tokens(self) -> int:
        return _lib.slm_get_total_tokens(self._ctx)

    def top_vocab(self, n: int = 20) -> list:
        import json
        res_ptr = _lib.slm_get_top_vocab(self._ctx, n)
        if not res_ptr:
            return []
        json_str = ctypes.string_at(res_ptr).decode("utf-8")
        _lib.slm_free_string(res_ptr)
        try:
            items = json.loads(json_str)
            return [(item["word"], item["count"]) for item in items]
        except Exception:
            return []

    def reset(self):
        _lib.slm_reset(self._ctx)
