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


BANNER = f"""
{C.CYAN}  ╔═══════════════════════════════════════════════════════╗{C.RESET}
{C.CYAN}  ║                                                       ║{C.RESET}
{C.CYAN}  ║   ███████╗██╗     ███╗   ███╗                         ║{C.RESET}
{C.CYAN}  ║   ██╔════╝██║     ████╗ ████║                         ║{C.RESET}
{C.CYAN}  ║   ███████╗██║     ██╔████╔██║                         ║{C.RESET}
{C.CYAN}  ║   ╚════██║██║     ██║╚██╔╝██║                         ║{C.RESET}
{C.CYAN}  ║   ███████║███████╗██║ ╚═╝ ██║                         ║{C.RESET}
{C.CYAN}  ║   ╚══════╝╚══════╝╚═╝     ╚═╝                         ║{C.RESET}
{C.CYAN}  ║                                                       ║{C.RESET}
{C.CYAN}  ║   {C.WHITE}Singular Language Model{C.CYAN}                            ║{C.RESET}
{C.CYAN}  ║   {C.GRAY}v{__version__} — Offline Frequency-Based AI{C.CYAN}               ║{C.RESET}
{C.CYAN}  ║   {C.GRAY}No internet. No cloud. Just local intelligence.{C.CYAN}    ║{C.RESET}
{C.CYAN}  ║                                                       ║{C.RESET}
{C.CYAN}  ╚═══════════════════════════════════════════════════════╝{C.RESET}
"""


def parse_input(line: str):
    """Parse a command line into command name and arguments."""
    try:
        parts = shlex.split(line)
    except ValueError:
        parts = line.strip().split()

    if not parts:
        return None, []

    return parts[0].lower(), parts[1:]


def run_interactive(engine: SLMEngine, trainer: SLMTrainer):
    """Run the interactive REPL shell."""
    print(BANNER)
    print(f"  {C.GRAY}Type 'help' for commands, 'exit' to quit.{C.RESET}\n")

    while True:
        try:
            line = input(f"  {C.MAGENTA}slm >{C.RESET} ")
        except (EOFError, KeyboardInterrupt):
            print(f"\n{C.GRAY}  Goodbye!{C.RESET}")
            break

        line = line.strip()
        if not line:
            continue

        if line.lower() in ("exit", "quit", "q"):
            print(f"{C.GRAY}  Goodbye!{C.RESET}")
            break

        command_name, args = parse_input(line)
        if command_name:
            execute_command(command_name, engine, trainer, args)
            print()  # Blank line between commands


def run_single(engine: SLMEngine, trainer: SLMTrainer, command_name: str, args: list):
    """Run a single command and exit."""
    execute_command(command_name, engine, trainer, args)


def main():
    """Main entry point."""
    # Enable ANSI colors on Windows
    if sys.platform == "win32":
        os.system("")  # Enables ANSI escape codes on Windows 10+

    engine = SLMEngine()
    trainer = SLMTrainer(engine)

    if len(sys.argv) > 1:
        # Single command mode: slm <command> [args...]
        command_name = sys.argv[1].lower()

        if command_name in ("--help", "-h"):
            command_name = "help"

        if command_name in ("--version", "-v"):
            command_name = "version"

        args = sys.argv[2:]
        run_single(engine, trainer, command_name, args)
    else:
        # Interactive REPL mode
        run_interactive(engine, trainer)


if __name__ == "__main__":
    main()
