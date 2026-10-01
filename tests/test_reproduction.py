import pandas as pd
import pytest

from scripts.verify_reproduction import assert_same_table


def test_numeric_tolerance_cannot_hide_a_different_game_id():
    expected = pd.DataFrame({"nbagameid": [22301181], "probability": [0.915311]})
    # A relative numeric tolerance would incorrectly accept a one-unit ID change.
    with pytest.raises(AssertionError):
        assert_same_table(expected, expected.assign(nbagameid=22301182), "predictions")
    assert_same_table(expected, expected.assign(probability=0.915311001), "predictions")
    with pytest.raises(AssertionError):
        assert_same_table(
            expected, expected.assign(probability=0.925311), "predictions"
        )
    with pytest.raises(AssertionError):
        assert_same_table(
            expected, expected[["probability", "nbagameid"]], "predictions"
        )
