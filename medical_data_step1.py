import argparse
import importlib.util
import tempfile
from pathlib import Path

import pandas as pd


DATASET_PATHS = {
    "hungarian": Path("input/processed.hungarian.data source.csv"),
    "va": Path("input/processed.va.data source.csv"),
    "switzerland": Path("input/processed.switzerland.data source.csv"),
}

DEFAULT_DATASET = "hungarian"
DEFAULT_CHARTS_DIR = Path("output/charts")
DEFAULT_PDF_REPORT = Path("output/heart_disease_analysis_report.pdf")
DEFAULT_CSV_EXPORT_DIR = Path("output/csv")

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
        bins=[0, 39, 55, 120],
        labels=["<40", "40-55", "56+"],
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


def filter_by_age_range(df: pd.DataFrame, min_age: int | None = None, max_age: int | None = None) -> pd.DataFrame:
    """Filter patients by an inclusive age range."""
    filtered = df.copy()
    if min_age is not None:
        filtered = filtered[filtered["age"] >= min_age]
    if max_age is not None:
        filtered = filtered[filtered["age"] <= max_age]
    return filtered


def filter_by_sex(df: pd.DataFrame, sex: str) -> pd.DataFrame:
    """Filter patients by sex label."""
    return df[df["sex_label"] == sex].copy()


def filter_by_diagnosis(df: pd.DataFrame, diagnosis: str, binary: bool = True) -> pd.DataFrame:
    """Filter by diagnosis label (binary or multiclass)."""
    diagnosis_column = "diagnosis_binary" if binary else "diagnosis"
    return df[df[diagnosis_column] == diagnosis].copy()


def filter_by_cholesterol_range(
    df: pd.DataFrame,
    min_cholesterol: float | None = None,
    max_cholesterol: float | None = None,
) -> pd.DataFrame:
    """Filter by inclusive cholesterol range."""
    filtered = df.copy()
    if min_cholesterol is not None:
        filtered = filtered[filtered["cholesterol"] >= min_cholesterol]
    if max_cholesterol is not None:
        filtered = filtered[filtered["cholesterol"] <= max_cholesterol]
    return filtered


def filter_by_blood_pressure_range(
    df: pd.DataFrame,
    min_bp: float | None = None,
    max_bp: float | None = None,
) -> pd.DataFrame:
    """Filter by inclusive resting blood pressure range."""
    filtered = df.copy()
    if min_bp is not None:
        filtered = filtered[filtered["trestbps"] >= min_bp]
    if max_bp is not None:
        filtered = filtered[filtered["trestbps"] <= max_bp]
    return filtered


def calculate_mean(df: pd.DataFrame, column: str) -> float:
    """Return mean for a numeric column."""
    return round(float(df[column].mean()), 2)


def calculate_median(df: pd.DataFrame, column: str) -> float:
    """Return median for a numeric column."""
    return round(float(df[column].median()), 2)


def group_by_sex(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate statistics grouped by sex."""
    return (
        df.groupby("sex_label", dropna=False)
        .agg(
            patient_count=("sex_label", "size"),
            average_age=("age", "mean"),
            average_cholesterol=("cholesterol", "mean"),
            average_resting_blood_pressure=("trestbps", "mean"),
            average_max_heart_rate=("maximum_heart_rate", "mean"),
        )
        .round(2)
        .reset_index()
    )


def group_by_age_group(df: pd.DataFrame, age_group_column: str = "age_group") -> pd.DataFrame:
    """Aggregate statistics grouped by age group."""
    return (
        df.groupby(age_group_column, dropna=False)
        .agg(
            patient_count=(age_group_column, "size"),
            average_age=("age", "mean"),
            average_cholesterol=("cholesterol", "mean"),
            average_resting_blood_pressure=("trestbps", "mean"),
            average_max_heart_rate=("maximum_heart_rate", "mean"),
        )
        .round(2)
        .reset_index()
    )


def group_by_disease_status(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate statistics grouped by binary disease status."""
    return (
        df.groupby("diagnosis_binary", dropna=False)
        .agg(
            patient_count=("diagnosis_binary", "size"),
            average_age=("age", "mean"),
            average_cholesterol=("cholesterol", "mean"),
            average_resting_blood_pressure=("trestbps", "mean"),
            average_max_heart_rate=("maximum_heart_rate", "mean"),
        )
        .round(2)
        .reset_index()
    )


def build_core_analysis_outputs(df: pd.DataFrame) -> dict[str, object]:
    """Build core pandas analysis outputs required before GUI work."""
    men_df = filter_by_sex(df, "male")
    women_df = filter_by_sex(df, "female")
    disease_df = filter_by_diagnosis(df, "disease", binary=True)
    no_disease_df = filter_by_diagnosis(df, "no disease", binary=True)

    return {
        "filter_examples": {
            "age_40_to_55_count": int(len(filter_by_age_range(df, min_age=40, max_age=55))),
            "male_count": int(len(men_df)),
            "disease_count": int(len(disease_df)),
            "cholesterol_200_to_240_count": int(len(filter_by_cholesterol_range(df, 200, 240))),
            "blood_pressure_120_to_140_count": int(len(filter_by_blood_pressure_range(df, 120, 140))),
        },
        "mean_and_median": {
            "mean_age": calculate_mean(df, "age"),
            "median_age": calculate_median(df, "age"),
            "mean_cholesterol": calculate_mean(df, "cholesterol"),
            "median_cholesterol": calculate_median(df, "cholesterol"),
        },
        "example_metrics": {
            "average_cholesterol_men": calculate_mean(men_df, "cholesterol"),
            "average_cholesterol_women": calculate_mean(women_df, "cholesterol"),
            "median_age_disease": calculate_median(disease_df, "age"),
            "median_age_no_disease": calculate_median(no_disease_df, "age"),
            "count_by_age_category": df["age_group"].value_counts(dropna=False).to_dict(),
        },
        "groupings": {
            "by_sex": group_by_sex(df),
            "by_age_group": group_by_age_group(df),
            "by_disease_status": group_by_disease_status(df),
        },
    }


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
    def comparison_agg(group_column: str) -> pd.DataFrame:
        return (
            df.groupby(group_column, dropna=False)
            .agg(
                patient_count=(group_column, "size"),
                average_age=("age", "mean"),
                average_cholesterol=("cholesterol", "mean"),
                average_resting_blood_pressure=("trestbps", "mean"),
                average_max_heart_rate=("maximum_heart_rate", "mean"),
            )
            .round(2)
            .reset_index()
        )

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

    comparison_df = df.copy()
    comparison_df["cholesterol_group"] = comparison_df["cholesterol"].apply(
        lambda c: "high cholesterol" if c >= 240 else "normal cholesterol"
    )
    required_comparisons = {
        "men_vs_women": comparison_agg("sex_label"),
        "younger_vs_older": group_by_age_group(comparison_df, age_group_column="age_group"),
        "disease_vs_no_disease": comparison_agg("diagnosis_binary"),
        "chest_pain_types": comparison_agg("chest_pain"),
        "high_vs_normal_cholesterol": comparison_df.groupby("cholesterol_group", dropna=False)
        .agg(
            patient_count=("cholesterol_group", "size"),
            average_age=("age", "mean"),
            average_cholesterol=("cholesterol", "mean"),
            average_resting_blood_pressure=("trestbps", "mean"),
            average_max_heart_rate=("maximum_heart_rate", "mean"),
        )
        .round(2)
        .reset_index(),
    }

    return {
        "by_sex": by_sex,
        "by_age_group": by_age_group,
        "by_chest_pain": by_chest_pain,
        "by_exercise_induced_angina": by_exercise_induced_angina,
        "sex_diagnosis_distribution_pct": sex_diagnosis_distribution,
        "age_group_diagnosis_distribution_pct": age_group_diagnosis_distribution,
        "chest_pain_diagnosis_distribution_pct": chest_pain_diagnosis_distribution,
        "exercise_angina_disease_distribution_pct": exercise_angina_disease_distribution,
        "required_comparisons": required_comparisons,
    }


def _save_histogram(df: pd.DataFrame, column: str, title: str, output_file: Path, bins: int = 15) -> None:
    """Render and save a histogram for a numeric column."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df[column].dropna(), bins=bins, color="#4C78A8", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(column.replace("_", " ").title())
    ax.set_ylabel("Patient count")
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def _save_scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str,
    title: str,
    output_file: Path,
) -> None:
    """Render and save a scatter plot colored by a categorical column."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, group_df in df.groupby(hue_col, dropna=False):
        ax.scatter(
            group_df[x_col],
            group_df[y_col],
            alpha=0.75,
            s=30,
            label=str(label),
        )
    ax.set_title(title)
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(y_col.replace("_", " ").title())
    ax.legend(title=hue_col.replace("_", " ").title())
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def _save_bar_chart(value_counts: pd.Series, title: str, x_label: str, output_file: Path) -> None:
    """Render and save a bar chart from a value-count series."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    value_counts.plot(kind="bar", ax=ax, color="#59A14F")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Patient count")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def generate_visualizations(df: pd.DataFrame, charts_dir: Path) -> list[Path]:
    """Generate six core visuals using the currently filtered dataset."""
    if importlib.util.find_spec("matplotlib") is None:
        print(
            "WARNING: matplotlib is not installed, skipping chart generation. "
            "Install dependencies from requirements.txt to enable visuals."
        )
        return []

    charts_dir.mkdir(parents=True, exist_ok=True)
    generated_files: list[Path] = []

    visualization_specs = [
        (
            "hist_age.png",
            lambda: _save_histogram(
                df, "age", "Age Distribution (Filtered Patients)", charts_dir / "hist_age.png"
            ),
        ),
        (
            "hist_resting_blood_pressure.png",
            lambda: _save_histogram(
                df,
                "trestbps",
                "Resting Blood Pressure Distribution (Filtered Patients)",
                charts_dir / "hist_resting_blood_pressure.png",
            ),
        ),
        (
            "hist_max_heart_rate.png",
            lambda: _save_histogram(
                df,
                "maximum_heart_rate",
                "Maximum Heart Rate Distribution (Filtered Patients)",
                charts_dir / "hist_max_heart_rate.png",
            ),
        ),
        (
            "scatter_age_vs_max_hr.png",
            lambda: _save_scatter_plot(
                df,
                "age",
                "maximum_heart_rate",
                "diagnosis_label",
                "Age vs Maximum Heart Rate (Filtered Patients)",
                charts_dir / "scatter_age_vs_max_hr.png",
            ),
        ),
        (
            "bar_diagnosis_distribution.png",
            lambda: _save_bar_chart(
                df["diagnosis_label"].value_counts(dropna=False),
                "Diagnosis Distribution (Filtered Patients)",
                "Diagnosis label",
                charts_dir / "bar_diagnosis_distribution.png",
            ),
        ),
        (
            "bar_chest_pain_distribution.png",
            lambda: _save_bar_chart(
                df["chest_pain"].value_counts(dropna=False),
                "Chest Pain Type Distribution (Filtered Patients)",
                "Chest pain type",
                charts_dir / "bar_chest_pain_distribution.png",
            ),
        ),
    ]

    for filename, render in visualization_specs:
        output_file = charts_dir / filename
        render()
        generated_files.append(output_file)

    return generated_files


def export_analysis_to_pdf(
    stats: dict[str, object],
    diagnosis_multiclass: pd.DataFrame,
    diagnosis_binary: pd.DataFrame,
    output_pdf: Path,
) -> None:
    """Export key analysis outputs to a PDF report."""
    if importlib.util.find_spec("matplotlib") is None:
        print(
            "WARNING: matplotlib is not installed, skipping PDF generation. "
            "Install dependencies from requirements.txt to enable PDF export."
        )
        return

    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_pdf) as pdf:
        # Page 1: summary stats.
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        summary_lines = [
            "Heart Disease Analysis Report",
            "",
            "Basic statistics:",
        ]
        summary_lines.extend([f"- {key}: {value}" for key, value in stats.items()])
        ax.text(
            0.02,
            0.98,
            "\n".join(summary_lines),
            va="top",
            ha="left",
            fontsize=10,
            family="monospace",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2: multiclass diagnosis table.
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis("off")
        ax.set_title("Diagnosis Distribution (Multiclass)", fontsize=12, pad=12)
        table = ax.table(
            cellText=diagnosis_multiclass.fillna("").values,
            colLabels=diagnosis_multiclass.columns,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.3)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 3: binary diagnosis table.
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis("off")
        ax.set_title("Diagnosis Distribution (Binary)", fontsize=12, pad=12)
        table = ax.table(
            cellText=diagnosis_binary.fillna("").values,
            colLabels=diagnosis_binary.columns,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.4)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def apply_sidebar_filters(
    df: pd.DataFrame,
    age_range: tuple[int, int],
    selected_sex: list[str],
    selected_diagnosis: list[str],
    selected_chest_pain: list[str],
    bp_range: tuple[int, int],
    max_hr_range: tuple[int, int],
    selected_exang: list[str],
    selected_restecg: list[str],
    selected_slope: list[str],
    selected_fbs: list[str],
    selected_thal: list[str],
) -> pd.DataFrame:
    """Apply all Streamlit sidebar filters with pandas operations."""
    filtered = df.copy()

    filtered = filtered[(filtered["age"] >= age_range[0]) & (filtered["age"] <= age_range[1])]
    filtered = filtered[(filtered["trestbps"] >= bp_range[0]) & (filtered["trestbps"] <= bp_range[1])]
    filtered = filtered[
        (filtered["maximum_heart_rate"] >= max_hr_range[0])
        & (filtered["maximum_heart_rate"] <= max_hr_range[1])
    ]

    categorical_filters = {
        "sex_label": selected_sex,
        "diagnosis": selected_diagnosis,
        "chest_pain": selected_chest_pain,
        "exercise_induced_angina_label": selected_exang,
        "restecg": selected_restecg,
        "slope_label": selected_slope,
        "fasting_blood_sugar_label": selected_fbs,
        "thal_label": selected_thal,
    }

    for column, selected_values in categorical_filters.items():
        if selected_values:
            filtered = filtered[filtered[column].isin(selected_values)]

    return filtered


def _build_pdf_bytes(
    stats: dict[str, object],
    diagnosis_multiclass: pd.DataFrame,
    diagnosis_binary: pd.DataFrame,
) -> bytes:
    """Create PDF report content as bytes for Streamlit download."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
        tmp_path = Path(tmp_pdf.name)
    try:
        export_analysis_to_pdf(stats, diagnosis_multiclass, diagnosis_binary, tmp_path)
        return tmp_path.read_bytes() if tmp_path.exists() else b""
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def export_analysis_to_csv(
    filtered_df: pd.DataFrame,
    comparison_tables: dict[str, pd.DataFrame | dict[str, pd.DataFrame]],
    diagnosis_tables: dict[str, pd.DataFrame],
    export_dir: Path,
) -> dict[str, Path]:
    """Export reusable analysis outputs to CSV files."""
    export_dir.mkdir(parents=True, exist_ok=True)

    exported_paths: dict[str, Path] = {}

    filtered_path = export_dir / "filtered_patients.csv"
    filtered_df.to_csv(filtered_path, index=False, sep=";")
    exported_paths["filtered_patients"] = filtered_path

    by_sex_path = export_dir / "group_comparison_by_sex.csv"
    comparison_tables["by_sex"].to_csv(by_sex_path, index=False, sep=";")
    exported_paths["group_comparison_by_sex"] = by_sex_path

    by_age_group_path = export_dir / "group_comparison_by_age_group.csv"
    comparison_tables["by_age_group"].to_csv(by_age_group_path, index=False, sep=";")
    exported_paths["group_comparison_by_age_group"] = by_age_group_path

    diagnosis_multiclass_path = export_dir / "diagnosis_distribution_multiclass.csv"
    diagnosis_tables["multiclass"].to_csv(diagnosis_multiclass_path, index=False, sep=";")
    exported_paths["diagnosis_distribution_multiclass"] = diagnosis_multiclass_path

    diagnosis_binary_path = export_dir / "diagnosis_distribution_binary.csv"
    diagnosis_tables["binary"].to_csv(diagnosis_binary_path, index=False, sep=";")
    exported_paths["diagnosis_distribution_binary"] = diagnosis_binary_path

    chest_pain_distribution_path = export_dir / "chest_pain_distribution.csv"
    comparison_tables["by_chest_pain"].to_csv(chest_pain_distribution_path, index=False, sep=";")
    exported_paths["chest_pain_distribution"] = chest_pain_distribution_path

    return exported_paths


def run_streamlit_app() -> None:
    """Render interactive Streamlit GUI for filtering and analysis."""
    import matplotlib.pyplot as plt
    import streamlit as st

    st.set_page_config(page_title="Heart Disease Analysis GUI", layout="wide")
    st.title("Heart Disease Analysis Dashboard")
    st.caption("Built with pandas + matplotlib + Streamlit")

    st.sidebar.header("Filters")
    dataset_key = st.sidebar.selectbox("Choose dataset", options=list(DATASET_PATHS.keys()), index=0)
    source_df = load_source_csv(DATASET_PATHS[dataset_key])
    cleaned_df = prepare_heart_disease_data(source_df)

    min_age, max_age = int(cleaned_df["age"].min()), int(cleaned_df["age"].max())
    min_bp, max_bp = int(cleaned_df["trestbps"].min()), int(cleaned_df["trestbps"].max())
    min_hr, max_hr = int(cleaned_df["maximum_heart_rate"].min()), int(cleaned_df["maximum_heart_rate"].max())

    age_range = st.sidebar.slider("Age range", min_value=min_age, max_value=max_age, value=(min_age, max_age))
    selected_sex = st.sidebar.multiselect("Sex", sorted(cleaned_df["sex_label"].dropna().unique().tolist()))
    selected_diagnosis = st.sidebar.multiselect("Diagnosis", sorted(cleaned_df["diagnosis"].dropna().unique().tolist()))
    selected_chest_pain = st.sidebar.multiselect(
        "Chest pain type", sorted(cleaned_df["chest_pain"].dropna().unique().tolist())
    )
    bp_range = st.sidebar.slider(
        "Resting blood pressure range",
        min_value=min_bp,
        max_value=max_bp,
        value=(min_bp, max_bp),
    )
    max_hr_range = st.sidebar.slider(
        "Maximum heart rate range",
        min_value=min_hr,
        max_value=max_hr,
        value=(min_hr, max_hr),
    )
    selected_exang = st.sidebar.multiselect(
        "Exercise-induced angina",
        sorted(cleaned_df["exercise_induced_angina_label"].dropna().unique().tolist()),
    )
    selected_restecg = st.sidebar.multiselect("Resting ECG", sorted(cleaned_df["restecg"].dropna().unique().tolist()))
    selected_slope = st.sidebar.multiselect("Slope", sorted(cleaned_df["slope_label"].dropna().unique().tolist()))

    with st.sidebar.expander("Optional extra filters"):
        selected_fbs = st.multiselect(
            "Fasting blood sugar",
            sorted(cleaned_df["fasting_blood_sugar_label"].dropna().unique().tolist()),
        )
        selected_thal = st.multiselect("Thal", sorted(cleaned_df["thal_label"].dropna().unique().tolist()))

    filtered_df = apply_sidebar_filters(
        cleaned_df,
        age_range=age_range,
        selected_sex=selected_sex,
        selected_diagnosis=selected_diagnosis,
        selected_chest_pain=selected_chest_pain,
        bp_range=bp_range,
        max_hr_range=max_hr_range,
        selected_exang=selected_exang,
        selected_restecg=selected_restecg,
        selected_slope=selected_slope,
        selected_fbs=selected_fbs,
        selected_thal=selected_thal,
    )

    st.subheader("Dataset preview (filtered)")
    st.write(f"Rows: {len(filtered_df)} / {len(cleaned_df)}")
    st.dataframe(filtered_df.head(200), use_container_width=True)

    if filtered_df.empty:
        st.warning("No rows match the selected filters. Adjust filters to continue analysis.")
        return

    st.subheader("Basic statistics")
    stats = basic_statistics(filtered_df)
    col1, col2, col3 = st.columns(3)
    col1.metric("Patients", stats["number_of_patients"])
    col2.metric("Average age", stats["average_age"])
    col3.metric("Median age", stats["median_age"])
    st.json(stats)

    st.subheader("Comparison of patient groups")
    comparisons = group_comparison_tables(filtered_df)
    st.markdown("**By sex**")
    st.dataframe(comparisons["by_sex"], use_container_width=True)
    st.markdown("**By age group**")
    st.dataframe(comparisons["by_age_group"], use_container_width=True)
    st.markdown("**By chest pain**")
    st.dataframe(comparisons["by_chest_pain"], use_container_width=True)
    st.markdown("**By exercise-induced angina**")
    st.dataframe(comparisons["by_exercise_induced_angina"], use_container_width=True)

    st.subheader("Visualization section")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    filtered_df["age"].plot(kind="hist", bins=15, ax=ax1, color="#4C78A8", edgecolor="white")
    ax1.set_title("Age distribution")
    st.pyplot(fig1)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    filtered_df["diagnosis_label"].value_counts(dropna=False).plot(kind="bar", ax=ax2, color="#59A14F")
    ax2.set_title("Diagnosis distribution")
    ax2.tick_params(axis="x", rotation=25)
    st.pyplot(fig2)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(6, 4))
    for label, group_df in filtered_df.groupby("diagnosis_label", dropna=False):
        ax3.scatter(group_df["age"], group_df["maximum_heart_rate"], s=30, alpha=0.75, label=str(label))
    ax3.set_xlabel("Age")
    ax3.set_ylabel("Maximum heart rate")
    ax3.set_title("Age vs maximum heart rate")
    ax3.legend()
    st.pyplot(fig3)
    plt.close(fig3)

    st.subheader("Export section")
    csv_bytes = filtered_df.to_csv(index=False, sep=";").encode("utf-8")
    st.download_button(
        "Download filtered CSV",
        data=csv_bytes,
        file_name=f"{dataset_key}_filtered.csv",
        mime="text/csv",
    )

    diagnosis_tables = diagnosis_distribution_tables(filtered_df)
    st.download_button(
        "Download group comparison by sex (CSV)",
        data=comparisons["by_sex"].to_csv(index=False, sep=";").encode("utf-8"),
        file_name=f"{dataset_key}_group_comparison_by_sex.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download group comparison by age group (CSV)",
        data=comparisons["by_age_group"].to_csv(index=False, sep=";").encode("utf-8"),
        file_name=f"{dataset_key}_group_comparison_by_age_group.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download diagnosis distribution (CSV)",
        data=diagnosis_tables["multiclass"].to_csv(index=False, sep=";").encode("utf-8"),
        file_name=f"{dataset_key}_diagnosis_distribution_multiclass.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download chest pain distribution (CSV)",
        data=comparisons["by_chest_pain"].to_csv(index=False, sep=";").encode("utf-8"),
        file_name=f"{dataset_key}_chest_pain_distribution.csv",
        mime="text/csv",
    )

    pdf_bytes = _build_pdf_bytes(stats, diagnosis_tables["multiclass"], diagnosis_tables["binary"])
    st.download_button(
        "Download PDF report",
        data=pdf_bytes,
        file_name=f"{dataset_key}_analysis_report.pdf",
        mime="application/pdf",
    )


def is_running_in_streamlit() -> bool:
    """Detect whether the script is executed via `streamlit run`."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


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
    parser.add_argument("--pdf-output", type=Path, default=DEFAULT_PDF_REPORT, help="Path for PDF report export")
    parser.add_argument(
        "--csv-export-dir",
        type=Path,
        default=DEFAULT_CSV_EXPORT_DIR,
        help="Directory for CSV analysis export files",
    )
    parser.add_argument(
        "--charts-dir",
        type=Path,
        default=DEFAULT_CHARTS_DIR,
        help="Directory where generated charts are saved",
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

    source_df = load_source_csv(input_path)
    cleaned_df = prepare_heart_disease_data(source_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_path, sep=";", index=False)

    filtered_analysis_df = apply_analysis_filters(
        cleaned_df,
        sex=args.filter_sex,
        age_group=args.filter_age_group,
        chest_pain=args.filter_chest_pain,
        diagnosis_binary=args.filter_diagnosis_binary,
    )

    overall_stats = basic_statistics(filtered_analysis_df)
    core_analysis_outputs = build_core_analysis_outputs(filtered_analysis_df)
    diagnosis_tables = diagnosis_distribution_tables(filtered_analysis_df)
    comparison_tables = group_comparison_tables(filtered_analysis_df)
    generated_charts = generate_visualizations(filtered_analysis_df, args.charts_dir)
    exported_csvs = export_analysis_to_csv(
        filtered_analysis_df,
        comparison_tables=comparison_tables,
        diagnosis_tables=diagnosis_tables,
        export_dir=args.csv_export_dir,
    )
    export_analysis_to_pdf(
        overall_stats,
        diagnosis_tables["multiclass"],
        diagnosis_tables["binary"],
        args.pdf_output,
    )

    print(f"Source rows loaded: {len(source_df)}")
    print(f"Rows after cleaning: {len(cleaned_df)}")
    print(f"Dataset key: {args.dataset}")
    print(f"Source file used: {input_path}")
    print(f"Cleaned file saved to: {output_path}")
    print(f"Rows after optional filters: {len(filtered_analysis_df)}")
    print(f"PDF report saved to: {args.pdf_output}")
    print(f"CSV export directory: {args.csv_export_dir}")

    print("\nBasic statistics:")
    print(overall_stats)

    print("\nCore analysis functions (plain pandas):")
    print("\nFilter examples:")
    print(core_analysis_outputs["filter_examples"])
    print("\nMean and median:")
    print(core_analysis_outputs["mean_and_median"])
    print("\nRequested example metrics:")
    print(core_analysis_outputs["example_metrics"])
    print("\nGrouping outputs:")
    for grouping_key, grouping_df in core_analysis_outputs["groupings"].items():
        print(f"\n{grouping_key}:")
        print(grouping_df.to_string(index=False))

    print("\nDiagnosis distribution (multiclass):")
    print(diagnosis_tables["multiclass"].to_string(index=False))

    print("\nDiagnosis distribution (binary):")
    print(diagnosis_tables["binary"].to_string(index=False))

    print("\nGroup comparison tables:")
    for table_key, table_value in comparison_tables.items():
        if isinstance(table_value, dict):
            print(f"\n{table_key}:")
            for nested_key, nested_df in table_value.items():
                print(f"\n{nested_key}:")
                print(nested_df.to_string(index=False))
            continue
        print(f"\n{table_key}:")
        print(table_value.to_string(index=False))

    print("\nGenerated charts:")
    for chart_path in generated_charts:
        print(f"- {chart_path}")

    print("\nExported CSV files:")
    for csv_key, csv_path in exported_csvs.items():
        print(f"- {csv_key}: {csv_path}")


if __name__ == "__main__":
    if is_running_in_streamlit():
        run_streamlit_app()
    else:
        run_cleaning_step()
