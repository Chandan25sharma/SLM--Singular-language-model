#!/usr/bin/env python3
"""
SLM CLI — Singular Language Model
Run: python slm_cli.py [command] [args...]
     python slm_cli.py              (interactive mode)
"""

import sys
import os

# Add python directory to path so slm package is importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "python"))

from slm.cli import main

if __name__ == "__main__":
    main()
