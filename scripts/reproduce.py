"""Rebuild the analysis from repository-local source files."""

import subprocess
import sys

from src.data import ROOT


def main():
    for module in [
        "build_data",
        "compare_models",
        "evaluate",
        "explore",
        "execute_notebooks",
    ]:
        subprocess.run(
            [sys.executable, "-m", "scripts." + module], cwd=ROOT, check=True
        )
    subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
