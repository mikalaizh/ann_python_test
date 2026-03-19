"""Step 1 data preparation for the medical analysis project.

What this script does:
1. Loads the raw CSV file with pandas.
2. Renames columns to readable names.
3. Replaces missing markers ('?') and handles missing values.
4. Makes the target variable understandable:
   - 0 -> "no disease"
   - >0 -> "disease"

Usage:
    python medical_data_step1.py

Optional arguments:
    --input   Path to raw CSV file.
    --output  Path to cleaned CSV file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path("input/processed.switzerland.data source.csv")
DEFAULT_OUTPUT = Path("input/processed.switzerland.data cleaned_step1.csv")


COLUMN_RENAME_MAP = {
    "cp": "chest_pain_type",
    "trestbps": "resting_blood_pressure",
    "chol": "cholesterol",
    "fbs": "fasting_blood_sugar_over_120",
    "restecg": "resting_ecg_result",
    "thalach": "max_heart_rate",
    "exang": "exercise_induced_angina",
    "num": "heart_disease_raw",
}


def load_raw_data(path: Path) -> pd.DataFrame:
    """Load source data with semicolon separator."""
    return pd.read_csv(path, sep=";", encoding="utf-8-sig")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply first-step cleaning and target engineering."""
    data = df.copy()

    # Rename columns to more readable names.
    data = data.rename(columns=COLUMN_RENAME_MAP)

    # Standardize missing values.
    data = data.replace("?", pd.NA)

    # Convert selected columns to numeric where possible.
    numeric_columns = [
        "age",
        "sex",
        "chest_pain_type",
        "resting_blood_pressure",
        "cholesterol",
        "fasting_blood_sugar_over_120",
        "resting_ecg_result",
        "max_heart_rate",
        "exercise_induced_angina",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "heart_disease_raw",
    ]
    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Basic missing-value handling.
    # - Numeric features: fill with median.
    # - Target: rows without target are removed.
    features = [c for c in data.columns if c != "heart_disease_raw"]
    numeric_features = data[features].select_dtypes(include="number").columns
    for col in numeric_features:
        data[col] = data[col].fillna(data[col].median())

    data = data.dropna(subset=["heart_disease_raw"]).copy()

    # Make target understandable: binary + label.
    data["heart_disease_binary"] = (data["heart_disease_raw"] > 0).astype(int)
    data["heart_disease_label"] = data["heart_disease_binary"].map(
        {0: "no disease", 1: "disease"}
    )

    # Optional readability mappings for core categorical fields.
    data["sex_label"] = data["sex"].map({0: "female", 1: "male"})
    data["exercise_induced_angina_label"] = data["exercise_induced_angina"].map(
        {0: "no", 1: "yes"}
    )

    cp_map = {
        1: "typical angina",
        2: "atypical angina",
        3: "non-anginal pain",
        4: "asymptomatic",
    }
    data["chest_pain_label"] = data["chest_pain_type"].map(cp_map)

    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 1 cleaning of medical CSV data")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Raw CSV path")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination for cleaned CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_df = load_raw_data(args.input)
    clean_df = clean_data(raw_df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(args.output, sep=";", index=False)

    print(f"Loaded rows: {len(raw_df)}")
    print(f"Saved cleaned rows: {len(clean_df)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
