"""Deprecated compatibility entry point for the historical plot evaluator.

New quantitative evaluation must use ``python -m evaluation`` or
``python evaluate.py``. This shim preserves old commands without letting the
legacy implementation occupy the repository root.
"""

from scripts.legacy.interactive_eval import main


if __name__ == "__main__":
    print(
        "NOTE: eval.py is the historical interactive plotter; "
        "use `python -m evaluation --help` for canonical metrics."
    )
    main()
