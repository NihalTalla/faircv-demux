"""Dataset validation with human-readable errors.

The goal: a user who uploads a bad CSV should read
  "Protected attribute 'gender' contains only one group. Fairness comparison
   cannot be performed."
not
  "IndexError: list index out of range".

Validation is purely descriptive — it never modifies the upload.
"""

from dataclasses import dataclass, field

import pandas as pd


class DatasetValidationError(Exception):
    """Raised when an uploaded dataset cannot be used for an audit."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


@dataclass
class ValidationReport:
    row_count: int
    column_count: int
    columns: list[str]
    numeric_columns: list[str]
    label_column: str
    protected: dict[str, dict[str, int]]      # attr -> {group: count}
    missing_values: dict[str, int]            # column -> count
    duplicate_rows: int
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "columns": self.columns,
            "numeric_columns": self.numeric_columns,
            "label_column": self.label_column,
            "protected": self.protected,
            "missing_values": self.missing_values,
            "duplicate_rows": self.duplicate_rows,
            "warnings": self.warnings,
        }


MIN_GROUP_ROWS = 20   # below this, metrics are noisy — warn, don't block


def _require_column(columns: list[str], col: str, role: str) -> None:
    if col not in columns:
        raise DatasetValidationError(
            f"Column '{col}' (used as {role}) does not exist in the uploaded file. "
            f"Available columns: {', '.join(columns)}."
        )


def validate_csv(
    path: str,
    label_column: str,
    protected_attributes: list[str],
    feature_columns: list[str] | None = None,
) -> ValidationReport:
    """Validate a CSV for auditing. Raises DatasetValidationError on problems."""
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pandas raises many parsers; report the first clearly
        raise DatasetValidationError(
            f"Could not read the file as CSV: {exc}. "
            "Export your file as UTF-8 CSV (comma-separated) and try again."
        ) from exc

    if df.empty:
        raise DatasetValidationError("The uploaded file contains no data rows.")

    columns = [str(c) for c in df.columns]
    _require_column(columns, label_column, "target/label column")
    for attr in protected_attributes:
        _require_column(columns, attr, "protected attribute")

    # Missing values in the columns we actually use
    missing: dict[str, int] = {}
    for col in [label_column, *protected_attributes]:
        n_missing = int(df[col].isna().sum())
        if n_missing:
            missing[col] = n_missing
    if missing:
        cols_str = ", ".join(f"'{c}' ({n} rows)" for c, n in missing.items())
        raise DatasetValidationError(
            f"Missing values found in required columns: {cols_str}. "
            "Fairness metrics need a label and a protected attribute for every row. "
            "Clean or impute the missing values and re-upload."
        )

    # Protected-attribute group sanity
    protected: dict[str, dict[str, int]] = {}
    for attr in protected_attributes:
        counts = df[attr].value_counts(dropna=True)
        groups = {str(k): int(v) for k, v in counts.items()}
        protected[attr] = groups
        if len(groups) < 2:
            raise DatasetValidationError(
                f"Protected attribute '{attr}' contains only one group "
                f"({', '.join(groups)}). Fairness comparison cannot be performed "
                "— at least two groups are required."
            )
        for g, n in groups.items():
            if n < MIN_GROUP_ROWS:
                raise DatasetValidationError(
                    f"Protected attribute '{attr}' group '{g}' has only {n} rows. "
                    "Fairness metrics and bootstrap confidence intervals need more "
                    f"data — please use at least {MIN_GROUP_ROWS} rows per group."
                )

    # Numeric features
    numeric_columns = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
    if feature_columns:
        for col in feature_columns:
            _require_column(columns, col, "feature")
            if col not in numeric_columns:
                raise DatasetValidationError(
                    f"Feature column '{col}' is not numeric. Model training needs "
                    "numeric features — encode categorical values first."
                )
    else:
        # Default: all numeric columns except the label and protected attrs
        excluded = {label_column, *protected_attributes}
        feature_columns = [c for c in numeric_columns if c not in excluded]
    if not feature_columns:
        raise DatasetValidationError(
            "No numeric feature columns found. The audit trains a logistic model "
            "on numeric features — provide at least one numeric feature column."
        )

    # Duplicates: informational only
    dupes = int(df.duplicated().sum())

    warnings: list[str] = []
    if dupes:
        warnings.append(f"File contains {dupes} duplicate row(s); duplicates are kept as-is.")
    label_dtype = df[label_column].dtype
    if not pd.api.types.is_numeric_dtype(label_dtype):
        raise DatasetValidationError(
            f"Target/label column '{label_column}' is not numeric "
            f"(dtype {label_dtype}). The audit binarises labels at the training "
            "median, which requires a numeric score."
        )

    return ValidationReport(
        row_count=int(len(df)),
        column_count=len(columns),
        columns=columns,
        numeric_columns=numeric_columns,
        label_column=label_column,
        protected=protected,
        missing_values={},
        duplicate_rows=dupes,
        warnings=warnings,
    )
