import hashlib

import pytest

from scripts.release import verify_files


def test_changed_or_missing_artifact_is_rejected(tmp_path):
    source = tmp_path / "prediction.csv"
    original = b"probability\n0.7\n"
    source.write_bytes(original)
    manifest = {"sha256": {source.name: hashlib.sha256(original).hexdigest()}}
    verify_files(manifest, tmp_path)
    source.write_text("probability\n0.9\n")
    with pytest.raises(ValueError, match="prediction.csv"):
        verify_files(manifest, tmp_path)
    source.unlink()
    with pytest.raises(ValueError, match="prediction.csv"):
        verify_files(manifest, tmp_path)
