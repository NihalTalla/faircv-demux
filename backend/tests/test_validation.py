"""Dataset validation tests — every failure must be a clear message."""

import numpy as np
import pandas as pd
import pytest

from app.validation import DatasetValidationError, validate_csv


def _write(tmp_path, df, name="d.csv"):
    p = tmp_path / name
    df.to_csv(p, index=False)
    return str(p)


def _valid_df(n=300):
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "score": rng.normal(0.5, 0.15, n).clip(0, 1),
        "f1": rng.normal(size=n),
        "gender": rng.choice(["M", "F"], n),
        "ethnicity": rng.choice(["A", "B", "C"], n),
    })


def test_valid_csv_passes(tmp_path):
    path = _write(tmp_path, _valid_df())
    report = validate_csv(path, "score", ["gender", "ethnicity"])
    assert report.row_count == 300
    assert report.column_count == 4
    assert set(report.protected) == {"gender", "ethnicity"}
    assert len(report.protected["ethnicity"]) == 3
    assert report.missing_values == {}


def test_missing_label_column(tmp_path):
    path = _write(tmp_path, _valid_df().drop(columns=["score"]))
    with pytest.raises(DatasetValidationError, match="does not exist"):
        validate_csv(path, "score", ["gender"])


def test_missing_protected_column(tmp_path):
    path = _write(tmp_path, _valid_df())
    with pytest.raises(DatasetValidationError, match="does not exist"):
        validate_csv(path, "score", ["age"])


def test_single_group_rejected(tmp_path):
    df = _valid_df()
    df["gender"] = "M"  # one group only
    path = _write(tmp_path, df)
    with pytest.raises(DatasetValidationError, match="only one group"):
        validate_csv(path, "score", ["gender"])


def test_missing_values_rejected(tmp_path):
    df = _valid_df()
    df.loc[df.index[:5], "score"] = np.nan
    path = _write(tmp_path, df)
    with pytest.raises(DatasetValidationError, match="Missing values"):
        validate_csv(path, "score", ["gender"])


def test_non_numeric_label_rejected(tmp_path):
    df = _valid_df()
    df["score"] = np.where(df.index % 3 == 0, "high", "low")
    path = _write(tmp_path, df)
    with pytest.raises(DatasetValidationError, match="not numeric"):
        validate_csv(path, "score", ["gender"])


def test_tiny_group_rejected(tmp_path):
    df = _valid_df()
    df.loc[df.index[:150], "gender"] = "R"  # a 150-row third group < min? no, 150 >= 20
    # instead: shrink a group below MIN_GROUP_ROWS
    df = pd.concat([df[df.gender == "M"].head(10),
                    df[df.gender == "F"]])
    path = _write(tmp_path, df)
    with pytest.raises(DatasetValidationError, match="only 10 rows"):
        validate_csv(path, "score", ["gender"])


def test_non_numeric_feature_rejected(tmp_path):
    df = _valid_df()
    df["f1"] = np.where(df.index % 3 == 0, "a", "b")
    path = _write(tmp_path, df)
    with pytest.raises(DatasetValidationError, match="not numeric"):
        validate_csv(path, "score", ["gender"], feature_columns=["f1"])


def test_no_features_rejected(tmp_path):
    df = _valid_df()[["score", "gender", "ethnicity"]]
    path = _write(tmp_path, df)
    with pytest.raises(DatasetValidationError, match="No numeric feature"):
        validate_csv(path, "score", ["gender"])
