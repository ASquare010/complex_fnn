"""Compatibility entry point for the single canonical native256 memory trainer.

Historical raw512 chat training is available in the archived source, not this CLI.
"""
from models.token_memory.training import *
sources = source_hashes

if __name__ == "__main__":
    main()
