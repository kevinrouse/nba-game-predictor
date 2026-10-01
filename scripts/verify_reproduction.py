"""Compare a full rebuild with the artifacts committed at HEAD.

Numerical comparisons allow small platform differences; identifiers, row order,
column names and the selected configuration must match. Serialized model bytes
are verified separately before rebuilding, because serialization is not portable.
"""

import gzip
import io
import json
import subprocess

import numpy as np
import pandas as pd

from src.data import ROOT


def committed_bytes(path):
    return subprocess.run(
        ["git", "show", f"HEAD:{path}"], cwd=ROOT, check=True, capture_output=True
    ).stdout


def main():
    names = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "HEAD", "artifacts/tables"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    checked = 0
    for name in names:
        if not name.endswith((".csv", ".csv.gz")):
            continue
        raw = committed_bytes(name)
        expected = pd.read_csv(
            io.BytesIO(gzip.decompress(raw) if name.endswith(".gz") else raw)
        )
        actual = pd.read_csv(ROOT / name)
        pd.testing.assert_frame_equal(
            expected, actual, check_exact=False, rtol=1e-7, atol=1e-8, obj=name
        )
        checked += 1
    name = "artifacts/models/selection.json"
    expected = json.loads(committed_bytes(name))
    actual = json.loads((ROOT / name).read_text())
    expected_loss = expected.pop("validation_log_loss")
    actual_loss = actual.pop("validation_log_loss")
    if actual != expected:
        raise ValueError("Reproduction changed the selected model or evaluation design")
    np.testing.assert_allclose(actual_loss, expected_loss, rtol=1e-7, atol=1e-8)
    print(f"Reproduced {checked} published tables and the original model selection")


if __name__ == "__main__":
    main()
