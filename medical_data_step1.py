import argparse
import re
import sqlite3
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path("input/processed.switzerland.data source.csv")
DEFAULT_OUTPUT = Path("input/processed.switzerland.data cleaned_step1.csv")
DEFAULT_DB = Path("input/heart_disease_analysis.db")
DEFAULT_TABLE = "heart_disease_cleaned"

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
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the source CSV file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path where the cleaned CSV should be saved",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help="Path to the SQLite database file",
    )
    parser.add_argument(
        "--table",
        default=DEFAULT_TABLE,
        help="Table name to create/load in the SQLite database",
    )
    parser.add_argument(
        "--skip-db",
        action="store_true",
        help="Skip loading the cleaned data into SQLite",
    )
    return parser.parse_args()


def validate_table_name(table_name: str) -> str:
    """Allow only simple SQL identifiers for destination table names."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table_name):
        raise ValueError(
            "Invalid table name. Use letters, numbers, and underscores only, "
            "starting with a letter or underscore."
        )
    return table_name


def write_dataframe_to_sqlite(df: pd.DataFrame, db_path: Path, table_name: str) -> int:
    """Create/replace the destination table and insert all rows."""
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

    return row_count


def read_dataframe_from_sqlite(db_path: Path, table_name: str) -> pd.DataFrame:
    """Read data back from SQLite so analysis can continue from the DB table."""
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)


def run_cleaning_step() -> None:
    args = get_cli_args()

    source_df = load_source_csv(args.input)
    cleaned_df = prepare_heart_disease_data(source_df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(args.output, sep=";", index=False)

    print(f"Source rows loaded: {len(source_df)}")
    print(f"Rows after cleaning: {len(cleaned_df)}")
    print(f"Cleaned file saved to: {args.output}")

    if args.skip_db:
        print("Skipped SQLite load step (--skip-db).")
        return

    table_name = validate_table_name(args.table)
    inserted_rows = write_dataframe_to_sqlite(cleaned_df, args.db, table_name)
    reloaded_df = read_dataframe_from_sqlite(args.db, table_name)

    print(f"Rows inserted into table '{table_name}': {inserted_rows}")
    print(f"Rows reloaded from database: {len(reloaded_df)}")
    print(f"SQLite database saved to: {args.db}")


if __name__ == "__main__":
    run_cleaning_step()
