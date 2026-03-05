"""
SLM CLI - Main entry point for the Singular Language Model.
Supports both single-command mode and interactive REPL.
"""

import os
import sys
import shlex
from .engine import SLMEngine
from .trainer import SLMTrainer
from .commands import execute_command, COMMANDS, C
from . import __version__


BANNER = f"""\
{C.CYAN}  ╔══════════════════════════════════════════════════════════╗{C.RESET}
{C.CYAN}  ║                                                          ║{C.RESET}
{C.CYAN}  ║   ███████╗██╗      ███╗   ███╗                          ║{C.RESET}
{C.CYAN}  ║   ██╔════╝██║      ████╗ ████║                          ║{C.RESET}
{C.CYAN}  ║   ███████╗██║      ██╔████╔██║                          ║{C.RESET}
{C.CYAN}  ║   ╚════██║██║      ██║╚██╔╝██║                          ║{C.RESET}
{C.CYAN}  ║   ███████║███████╗ ██║ ╚═╝ ██║                          ║{C.RESET}
{C.CYAN}  ║   ╚══════╝╚══════╝ ╚═╝     ╚═╝                          ║{C.RESET}
{C.CYAN}  ║                                                          ║{C.RESET}
{C.CYAN}  ║   {C.WHITE}Singular Language Model{C.CYAN}  {C.GRAY}v{__version__}{C.CYAN}                       ║{C.RESET}
{C.CYAN}  ║   {C.GRAY}Offline · Frequency-Based · No Internet{C.CYAN}              ║{C.RESET}
{C.CYAN}  ║                                                          ║{C.RESET}
{C.CYAN}  ╚══════════════════════════════════════════════════════════╝{C.RESET}"""


def _status_bar(engine: SLMEngine) -> str:
    """Return a compact status line for the REPL prompt."""
    if engine.is_trained:
        return f"{C.GREEN}●{C.RESET} {C.GRAY}{engine.vocab_size():,} words{C.RESET}"
    else:
        return f"{C.YELLOW}○ untrained{C.RESET}"


def _prompt(engine: SLMEngine) -> str:
    return f"  {C.MAGENTA}slm{C.RESET} [{_status_bar(engine)}] {C.GRAY}›{C.RESET} "


def parse_input(line: str):
    """Parse a command line into command name and arguments."""
    try:
        parts = shlex.split(line)
    except ValueError:
        parts = line.strip().split()

    if not parts:
        return None, []

    return parts[0].lower(), parts[1:]


def _maybe_autoload(engine: SLMEngine, trainer: SLMTrainer):
    """
    If models/ directory has a single .slm file, offer to auto-load it.
    Called once on startup.
    """
    models_dir = "models"
    if not os.path.isdir(models_dir):
        return

    slm_files = sorted([
        f for f in os.listdir(models_dir)
        if f.endswith(".slm") and os.path.getsize(os.path.join(models_dir, f)) > 0
    ])
    if len(slm_files) == 1:
        path = os.path.join(models_dir, slm_files[0])
        print(f"  {C.CYAN}Auto-loading saved model: {slm_files[0]}{C.RESET}")
        if engine.load_model(path):
            print(f"  {C.GREEN}✓ Ready  —  {engine.vocab_size():,} words loaded{C.RESET}\n")
        else:
            print(f"  {C.YELLOW}⚠ Could not load {slm_files[0]}{C.RESET}\n")
    elif len(slm_files) > 1:
        print(f"  {C.GRAY}Saved models found in models/. Use 'load models/<name>.slm' to load one.{C.RESET}\n")


def run_interactive(engine: SLMEngine, trainer: SLMTrainer):
    """Run the interactive REPL shell."""
    if sys.platform == "win32":
        os.system("")  # Enable ANSI on Windows

    print(BANNER)
    print()

    _maybe_autoload(engine, trainer)

    if not engine.is_trained:
        print(f"  {C.YELLOW}No model loaded.{C.RESET}  Run {C.CYAN}quickstart{C.RESET} to train instantly, or {C.CYAN}help{C.RESET} for all commands.")
        print()

    while True:
        try:
            line = input(_prompt(engine))
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {C.GRAY}Goodbye!{C.RESET}\n")
            break

        line = line.strip()
        if not line:
            continue

        if line.lower() in ("exit", "quit", "q", "bye"):
            print(f"  {C.GRAY}Goodbye!{C.RESET}\n")
            break

        command_name, args = parse_input(line)
        if command_name:
            execute_command(command_name, engine, trainer, args)


def run_single(engine: SLMEngine, trainer: SLMTrainer, command_name: str, args: list):
    """Run a single command and exit."""
    if sys.platform == "win32":
        os.system("")
    execute_command(command_name, engine, trainer, args)


def main():
    """Main entry point."""
    engine = SLMEngine()
    trainer = SLMTrainer(engine)

    if len(sys.argv) > 1:
        command_name = sys.argv[1].lower()

        if command_name in ("--help", "-h"):
            command_name = "help"
        elif command_name in ("--version", "-v"):
            command_name = "version"

        args = sys.argv[2:]
        run_single(engine, trainer, command_name, args)
    else:
        run_interactive(engine, trainer)


if __name__ == "__main__":
    main()
