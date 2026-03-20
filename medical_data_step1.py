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

    # Drop rows missing mandatory fields for analysis.
    data = data.dropna(subset=["resting_blood_pressure", "diagnosis_raw", "age", "max_heart_rate", "oldpeak"]).copy()

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
        3: "severity 3 (serious)",
        4: "severity 4 (critical)",
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

    # Analysis-friendly aliases requested by app logic.
    data["chest_pain"] = data["chest_pain_label"]
    data["trestbps"] = data["resting_blood_pressure"]
    data["restecg"] = data["resting_ecg_label"]
    data["maximum_heart_rate"] = data["max_heart_rate"]
    data["diagnosis"] = data["severity_label"]

    # Derived analysis fields.
    data["diagnosis_binary"] = data["diagnosis"].apply(
        lambda x: "disease" if x != "no disease" else "no disease"
    )
    data["age_group"] = pd.cut(
        data["age"],
        bins=[0, 44, 54, 64, 120],
        labels=["<45", "45-54", "55-64", "65+"],
    )
    severity_map = {
        "no disease": 0,
        "severity 1 (mild)": 1,
        "severity 2 (moderate)": 2,
        "severity 3 (serious)": 3,
        "severity 4 (critical)": 4,
    }
    data["severity_code"] = data["diagnosis"].map(severity_map)

    return data


def apply_analysis_filters(
    df: pd.DataFrame,
    sex: str | None = None,
    age_group: str | None = None,
    chest_pain: str | None = None,
    diagnosis_binary: str | None = None,
) -> pd.DataFrame:
    """Apply optional filters before computing analysis outputs."""
    filtered = df.copy()
    filters = {
        "sex_label": sex,
        "age_group": age_group,
        "chest_pain": chest_pain,
        "diagnosis_binary": diagnosis_binary,
    }
    for column, value in filters.items():
        if value is not None:
            filtered = filtered[filtered[column] == value]
    return filtered


def basic_statistics(df: pd.DataFrame) -> dict[str, object]:
    """General characteristics of the dataset."""
    return {
        "number_of_patients": int(len(df)),
        "average_age": round(float(df["age"].mean()), 2),
        "median_age": round(float(df["age"].median()), 2),
        "average_resting_blood_pressure": round(float(df["trestbps"].mean()), 2),
        "average_maximum_heart_rate": round(float(df["maximum_heart_rate"].mean()), 2),
        "average_oldpeak": round(float(df["oldpeak"].mean()), 2),
        "count_by_sex": df["sex_label"].value_counts(dropna=False).to_dict(),
        "count_by_chest_pain_type": df["chest_pain"].value_counts(dropna=False).to_dict(),
        "count_by_diagnosis": df["diagnosis"].value_counts(dropna=False).to_dict(),
        "count_by_exercise_induced_angina": df["exercise_induced_angina_label"].value_counts(dropna=False).to_dict(),
    }


def diagnosis_distribution_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Multiclass and binary diagnosis summaries."""
    multiclass = (
        df.groupby("diagnosis", dropna=False)
        .agg(
            patient_count=("diagnosis", "size"),
            average_age=("age", "mean"),
            average_trestbps=("trestbps", "mean"),
            average_maximum_heart_rate=("maximum_heart_rate", "mean"),
            average_oldpeak=("oldpeak", "mean"),
            severity_code=("severity_code", "first"),
        )
        .sort_values("severity_code")
        .drop(columns=["severity_code"])
        .round(2)
        .reset_index()
    )

    binary = (
        df.groupby("diagnosis_binary", dropna=False)
        .agg(
            patient_count=("diagnosis_binary", "size"),
            average_age=("age", "mean"),
            average_trestbps=("trestbps", "mean"),
            average_maximum_heart_rate=("maximum_heart_rate", "mean"),
            average_oldpeak=("oldpeak", "mean"),
        )
        .round(2)
        .reset_index()
    )

    return {"multiclass": multiclass, "binary": binary}


def group_comparison_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Patient-group comparison tables required by the app."""
    by_sex = (
        df.groupby("sex_label", dropna=False)
        .agg(
            patient_count=("sex_label", "size"),
            average_age=("age", "mean"),
            average_trestbps=("trestbps", "mean"),
            average_maximum_heart_rate=("maximum_heart_rate", "mean"),
            average_oldpeak=("oldpeak", "mean"),
            exercise_induced_angina_yes_pct=(
                "exercise_induced_angina",
                lambda s: (s == 1).mean() * 100,
            ),
        )
        .round(2)
        .reset_index()
    )

    by_age_group = (
        df.groupby("age_group", dropna=False)
        .agg(
            patient_count=("age_group", "size"),
            average_trestbps=("trestbps", "mean"),
            average_maximum_heart_rate=("maximum_heart_rate", "mean"),
            average_oldpeak=("oldpeak", "mean"),
            asymptomatic_chest_pain_pct=("chest_pain", lambda s: (s == "asymptomatic").mean() * 100),
            exercise_induced_angina_yes_pct=(
                "exercise_induced_angina",
                lambda s: (s == 1).mean() * 100,
            ),
        )
        .round(2)
        .reset_index()
    )

    by_chest_pain = (
        df.groupby("chest_pain", dropna=False)
        .agg(
            patient_count=("chest_pain", "size"),
            average_age=("age", "mean"),
            average_maximum_heart_rate=("maximum_heart_rate", "mean"),
            average_oldpeak=("oldpeak", "mean"),
        )
        .round(2)
        .reset_index()
    )

    by_exercise_induced_angina = (
        df.groupby("exercise_induced_angina_label", dropna=False)
        .agg(
            patient_count=("exercise_induced_angina_label", "size"),
            average_age=("age", "mean"),
            average_trestbps=("trestbps", "mean"),
            average_maximum_heart_rate=("maximum_heart_rate", "mean"),
            average_oldpeak=("oldpeak", "mean"),
            disease_pct=("diagnosis_binary", lambda s: (s == "disease").mean() * 100),
        )
        .round(2)
        .reset_index()
    )

    sex_diagnosis_distribution = pd.crosstab(
        df["sex_label"], df["diagnosis"], normalize="index"
    ).mul(100).round(2).reset_index()
    age_group_diagnosis_distribution = pd.crosstab(
        df["age_group"], df["diagnosis"], normalize="index"
    ).mul(100).round(2).reset_index()
    chest_pain_diagnosis_distribution = pd.crosstab(
        df["chest_pain"], df["diagnosis"], normalize="index"
    ).mul(100).round(2).reset_index()
    exercise_angina_disease_distribution = pd.crosstab(
        df["exercise_induced_angina_label"], df["diagnosis_binary"], normalize="index"
    ).mul(100).round(2).reset_index()

    return {
        "by_sex": by_sex,
        "by_age_group": by_age_group,
        "by_chest_pain": by_chest_pain,
        "by_exercise_induced_angina": by_exercise_induced_angina,
        "sex_diagnosis_distribution_pct": sex_diagnosis_distribution,
        "age_group_diagnosis_distribution_pct": age_group_diagnosis_distribution,
        "chest_pain_diagnosis_distribution_pct": chest_pain_diagnosis_distribution,
        "exercise_angina_disease_distribution_pct": exercise_angina_disease_distribution,
    }


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
    parser.add_argument("--filter-sex", type=str, default=None, help="Optional analysis filter")
    parser.add_argument("--filter-age-group", type=str, default=None, help="Optional analysis filter")
    parser.add_argument("--filter-chest-pain", type=str, default=None, help="Optional analysis filter")
    parser.add_argument(
        "--filter-diagnosis-binary",
        type=str,
        default=None,
        help="Optional analysis filter (disease/no disease)",
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
    filtered_analysis_df = apply_analysis_filters(
        analysis_df,
        sex=args.filter_sex,
        age_group=args.filter_age_group,
        chest_pain=args.filter_chest_pain,
        diagnosis_binary=args.filter_diagnosis_binary,
    )

    overall_stats = basic_statistics(filtered_analysis_df)
    diagnosis_tables = diagnosis_distribution_tables(filtered_analysis_df)
    comparison_tables = group_comparison_tables(filtered_analysis_df)

    print(f"Source rows loaded: {len(source_df)}")
    print(f"Rows after cleaning: {len(cleaned_df)}")
    print(f"Dataset key: {args.dataset}")
    print(f"Source file used: {input_path}")
    print(f"Cleaned file saved to: {output_path}")
    print(f"Cleaned data loaded to table: {table_name}")
    print(f"Database path: {args.db_path}")
    print(f"Rows available for analysis from DB: {len(analysis_df)}")
    print(f"Rows after optional filters: {len(filtered_analysis_df)}")

    print("\nBasic statistics:")
    print(overall_stats)

    print("\nDiagnosis distribution (multiclass):")
    print(diagnosis_tables["multiclass"].to_string(index=False))

    print("\nDiagnosis distribution (binary):")
    print(diagnosis_tables["binary"].to_string(index=False))

    print("\nGroup comparison tables:")
    for table_key, table_value in comparison_tables.items():
        print(f"\n{table_key}:")
        print(table_value.to_string(index=False))


if __name__ == "__main__":
    run_cleaning_step()
