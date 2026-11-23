"""Compatibility CLI for password/no-password/control evaluation.

Run from the repository root. See password_locking/README.md for CLI arguments.
"""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from password_locking.evaluate import main
from password_locking.scoring import analyze_file, compare_math_answers, extract_final_answer

if __name__ == "__main__":
    main()
