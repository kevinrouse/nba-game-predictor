"""Execute each notebook in a fresh kernel and save its displayed results."""

import nbformat
from nbclient import NotebookClient

from src.data import ROOT


def main():
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        NotebookClient(
            notebook,
            timeout=300,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        ).execute()
        nbformat.validate(notebook)
        nbformat.write(notebook, path)
        print(f"Executed {path.name}", flush=True)


if __name__ == "__main__":
    main()
