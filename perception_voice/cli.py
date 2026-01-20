"""
perception-voice CLI

Entry point for the perception-voice command.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from perception_voice import __version__
from perception_voice.client import EXIT_ERROR, EXIT_USAGE, client_get, client_set
from perception_voice.config import Config
from perception_voice.server import run_server


def setup_logging(verbose: bool = False) -> None:
    """Configure logging"""
    level = logging.DEBUG if verbose else logging.INFO
    format_str = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    
    logging.basicConfig(
        level=level,
        format=format_str,
        stream=sys.stdout,
    )
    
    # Reduce noise from third-party libraries
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        prog="perception-voice",
        description="System-level speech-to-text service",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"perception-voice {__version__}",
    )
    parser.add_argument(
        "-c", "--config",
        type=Path,
        help="Path to config file (default: config.yml in current directory)",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # serve command
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the server daemon",
    )
    serve_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    # client command
    client_parser = subparsers.add_parser(
        "client",
        help="Client commands",
    )
    client_subparsers = client_parser.add_subparsers(
        dest="client_command",
        help="Client subcommands",
    )
    
    # client set
    set_parser = client_subparsers.add_parser(
        "set",
        help="Set read marker to now",
    )
    set_parser.add_argument(
        "uid",
        help="Unique client identifier",
    )
    
    # client get
    get_parser = client_subparsers.add_parser(
        "get",
        help="Get transcriptions since read marker",
    )
    get_parser.add_argument(
        "uid",
        help="Unique client identifier",
    )
    
    return parser


def main(args: Optional[List[str]] = None) -> int:
    """Main entry point"""
    parser = create_parser()
    parsed = parser.parse_args(args)
    
    if not parsed.command:
        parser.print_help()
        return EXIT_USAGE
    
    # Load config
    config = Config.load(parsed.config)
    
    if parsed.command == "serve":
        setup_logging(verbose=parsed.verbose)
        run_server(config, verbose=parsed.verbose)
        return 0
    
    elif parsed.command == "client":
        if not parsed.client_command:
            parser.parse_args(["client", "--help"])
            return EXIT_USAGE
        
        # Minimal logging for client
        logging.basicConfig(
            level=logging.ERROR,
            format="%(message)s",
            stream=sys.stderr,
        )
        
        if parsed.client_command == "set":
            return client_set(config, parsed.uid)
        
        elif parsed.client_command == "get":
            return client_get(config, parsed.uid)
        
        else:
            print(f"Unknown client command: {parsed.client_command}", file=sys.stderr)
            return EXIT_USAGE
    
    else:
        print(f"Unknown command: {parsed.command}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
