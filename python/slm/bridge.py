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
        
        _lib.slm_generate.restype = ctypes.c_char_p
        _lib.slm_generate.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_double]
        
        _lib.slm_predict_next.restype = ctypes.c_char_p
        _lib.slm_predict_next.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        
        _lib.slm_get_vocab_size.restype = ctypes.c_int
        _lib.slm_get_vocab_size.argtypes = [ctypes.c_void_p]
        
        _lib.slm_get_total_tokens.restype = ctypes.c_int
        _lib.slm_get_total_tokens.argtypes = [ctypes.c_void_p]
        
        _lib.slm_free_string.restype = None
        _lib.slm_free_string.argtypes = [ctypes.c_char_p]
        
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
