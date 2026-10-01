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
        "story_analysis",
        "robustness",
        "execute_notebooks",
    ]:
        subprocess.run(
            [sys.executable, "-m", "scripts." + module], cwd=ROOT, check=True
        )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.export_forecast",
            "22301181",
            "--output",
            "artifacts/examples/detroit-at-dallas.json",
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
