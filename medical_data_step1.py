import argparse
import sqlite3
from pathlib import Path

import pandas as pd


DATASET_PATHS = {
    "hungarian": Path("input/processed.hungarian.data source.csv"),
    "va": Path("input/processed.va.data source.csv"),
    "switzerland": Path("input/processed.switzerland.data source.csv"),
}

DEFAULT_DATASET = "switzerland"
DEFAULT_DB = Path("input/heart_disease_analysis.db")

COLUMN_RENAME_MAP = {
    "cp": "chest_pain_type",
    "trestbps": "resting_blood_pressure",
    "chol": "cholesterol",
    "fbs": "fasting_blood_sugar_over_120",
    "restecg": "resting_ecg_result",
    "thalach": "max_heart_rate",
    "exang": "exercise_induced_angina",
    "num": "diagnosis_raw",
}


def load_source_csv(file_path: Path) -> pd.DataFrame:
    """Read the exported source file."""
    return pd.read_csv(file_path, sep=";", encoding="utf-8-sig")


def prepare_heart_disease_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw heart-disease dataset and add readable diagnosis fields."""
    data = df.copy()

    # Rename short source columns.
    data = data.rename(columns=COLUMN_RENAME_MAP)

    # Convert "?" to missing.
    data = data.replace("?", pd.NA)

    # Force numeric columns.
    columns_to_convert = [
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
        "diagnosis_raw",
    ]

    for column in columns_to_convert:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    # Drop rows with too many missing values.
    data = data[data.isna().sum(axis=1) <= 5].copy()

    # Drop rows missing blood pressure.
    data = data.dropna(subset=["resting_blood_pressure"]).copy()

    # Drop rows missing diagnosis.
    data = data.dropna(subset=["diagnosis_raw"]).copy()

    # Keep raw and binary targets.
    data["has_disease"] = (data["diagnosis_raw"] > 0).astype(int)
    data["diagnosis_label"] = data["has_disease"].map({
        0: "no disease",
        1: "disease",
    })
    data["severity_label"] = data["diagnosis_raw"].map({
        0: "no disease",
        1: "severity 1 (mild)",
        2: "severity 2 (moderate)",
        3: "severity 3 (high)",
        4: "severity 4 (very high)",
    })

    # Add readable labels.
    data["sex_label"] = data["sex"].map({
        0: "female",
        1: "male",
    })

    data["fasting_blood_sugar_label"] = data["fasting_blood_sugar_over_120"].map({
        0: "<= 120 mg/dl",
        1: "> 120 mg/dl",
    })

    data["resting_ecg_label"] = data["resting_ecg_result"].map({
        0: "normal",
        1: "ST-T wave abnormality",
        2: "left ventricular hypertrophy",
    })

    data["exercise_induced_angina_label"] = data["exercise_induced_angina"].map({
        0: "no",
        1: "yes",
    })

    chest_pain_map = {
        1: "typical angina",
        2: "atypical angina",
        3: "non-anginal pain",
        4: "asymptomatic",
    }
    data["chest_pain_label"] = data["chest_pain_type"].map(chest_pain_map)

    data["slope_label"] = data["slope"].map({
        1: "upsloping",
        2: "flat",
        3: "downsloping",
    })

    data["thal_label"] = data["thal"].map({
        3: "normal",
        6: "fixed defect",
        7: "reversible defect",
    })

    return data


def get_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean the first version of the heart disease CSV file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=sorted(DATASET_PATHS.keys()),
        default=DEFAULT_DATASET,
        help="Dataset key used to resolve defaults for input/output/table",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to the source CSV file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path where the cleaned CSV should be saved",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB,
        help="SQLite database path",
    )
    parser.add_argument(
        "--table-name",
        type=str,
        default=None,
        help="Table name for cleaned data",
    )
    return parser.parse_args()


def load_dataframe_to_sqlite(df: pd.DataFrame, db_path: Path, table_name: str) -> None:
    """Connect to SQLite and load the cleaned dataframe into a table."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        df.to_sql(table_name, connection, if_exists="replace", index=False)


def read_back_sqlite_table(db_path: Path, table_name: str) -> pd.DataFrame:
    """Reconnect to SQLite and read the table back for analysis."""
    with sqlite3.connect(db_path) as connection:
        return pd.read_sql_query(f"SELECT * FROM {table_name}", connection)


def run_cleaning_step() -> None:
    args = get_cli_args()

    input_path = args.input or DATASET_PATHS[args.dataset]
    if not input_path.exists():
        available_paths = "\n".join(
            [f"- {name}: {path}" for name, path in DATASET_PATHS.items()]
        )
        raise FileNotFoundError(
            f"Input file not found: {input_path}\n"
            "Use --input to provide a path or place a dataset file in one of:\n"
            f"{available_paths}"
        )

    output_path = (
        args.output
        or input_path.with_name(input_path.name.replace(" source.csv", " cleaned_step1.csv"))
    )
    table_name = args.table_name or f"{args.dataset}_cleaned_step1"

    source_df = load_source_csv(input_path)
    cleaned_df = prepare_heart_disease_data(source_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_path, sep=";", index=False)

    load_dataframe_to_sqlite(cleaned_df, args.db_path, table_name)
    analysis_df = read_back_sqlite_table(args.db_path, table_name)

    print(f"Source rows loaded: {len(source_df)}")
    print(f"Rows after cleaning: {len(cleaned_df)}")
    print(f"Dataset key: {args.dataset}")
    print(f"Source file used: {input_path}")
    print(f"Cleaned file saved to: {output_path}")
    print(f"Cleaned data loaded to table: {table_name}")
    print(f"Database path: {args.db_path}")
    print(f"Rows available for analysis from DB: {len(analysis_df)}")


if __name__ == "__main__":
    run_cleaning_step()
