"""Score a saved JSON/JSONL trace file using the shared grader.

Run from the repository root. See password_locking/README.md for CLI arguments.
"""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from password_locking.scoring import main
from password_locking.scoring import analyze_file, compare_math_answers, extract_final_answer

if __name__ == "__main__":
    main()
