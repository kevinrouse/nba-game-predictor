"""Compare a full rebuild with the artifacts committed at HEAD.

Numerical comparisons allow small platform differences; identifiers, row order,
column names and the selected configuration must match. Serialized model bytes
are verified separately before rebuilding, because serialization is not portable.
"""

import gzip
import io
import json
import platform
import subprocess

import numpy as np
import pandas as pd

from src.data import ROOT


def committed_bytes(path):
    return subprocess.run(
        ["git", "show", f"HEAD:{path}"], cwd=ROOT, check=True, capture_output=True
    ).stdout


def assert_same_table(expected, actual, name):
    """Require exact identity/count fields; tolerate only floating-point arithmetic."""
    pd.testing.assert_index_equal(
        expected.columns, actual.columns, exact=True, obj=name
    )
    numeric = expected.select_dtypes(include="floating").columns
    exact = expected.columns.difference(numeric, sort=False)
    pd.testing.assert_frame_equal(
        expected[exact], actual[exact], check_exact=True, obj=name
    )
    pd.testing.assert_frame_equal(
        expected[numeric],
        actual[numeric],
        check_exact=False,
        rtol=1e-7,
        atol=1e-8,
        obj=name,
    )


def main():
    names = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "HEAD", "artifacts/tables"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    checked = 0
    failures, differences = [], []
    for name in names:
        if not name.endswith((".csv", ".csv.gz")):
            continue
        raw = committed_bytes(name)
        expected = pd.read_csv(
            io.BytesIO(gzip.decompress(raw) if name.endswith(".gz") else raw)
        )
        actual = pd.read_csv(ROOT / name)
        try:
            assert_same_table(expected, actual, name)
        except AssertionError as error:
            failures.append(name)
            print(str(error).splitlines()[0], flush=True)
            if expected.shape == actual.shape and expected.columns.equals(
                actual.columns
            ):
                for column in expected.select_dtypes(include="number"):
                    different = ~np.isclose(
                        expected[column],
                        actual[column],
                        rtol=1e-7,
                        atol=1e-8,
                        equal_nan=True,
                    )
                    if not different.any():
                        continue
                    group = (
                        "name"
                        if "name" in expected
                        else "model"
                        if "model" in expected
                        else None
                    )
                    delta = (expected[column] - actual[column]).abs()
                    detail = pd.DataFrame(
                        {
                            "group": expected[group] if group else "all",
                            "difference": delta,
                            "changed": different,
                        }
                    )
                    for label, rows in detail[detail.changed].groupby("group"):
                        differences.append(
                            dict(
                                file=name,
                                column=column,
                                group=label,
                                rows=len(rows),
                                max_absolute_difference=rows.difference.max(),
                            )
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
    if failures:
        directory = ROOT / "artifacts/checks"
        directory.mkdir(exist_ok=True)
        detail = pd.DataFrame(differences)
        detail.to_csv(directory / "reproduction_differences.csv", index=False)
        print(detail.to_string(index=False), flush=True)
        print(
            f"Runtime: {platform.system()} {platform.machine()} Python {platform.python_version()}",
            flush=True,
        )
        raise AssertionError(
            f"Reproduction differed in {len(failures)} tables; see artifacts/checks/reproduction_differences.csv"
        )
    print(f"Reproduced {checked} published tables and the original model selection")


if __name__ == "__main__":
    main()
