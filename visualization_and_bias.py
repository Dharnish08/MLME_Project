from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Project paths

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures_initial_experiments"
TABLES_DIR = RESULTS_DIR / "tables_initial_experiments"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Constants

RECIPE_COLUMNS = ["T", "pH", "F1", "F2", "F3"]
NORMALIZED_COLUMNS = ["T_norm", "pH_norm", "F1_norm", "F2_norm", "F3_norm"]

VARIABLE_RANGES = {
    "T": (20.0, 60.0),
    "pH": (3.0, 9.5),
    "F1": (0.0, 2.0),
    "F2": (0.0, 2.0),
    "F3": (0.0, 2.0),
}

SCALE_MAP = {
    0: "micro",
    1: "bench",
    2: "pilot",
}


# Utility functions

def find_latest_log_file() -> Path:
    """Find the newest optimization log inside the results folder."""
    log_files = sorted(RESULTS_DIR.glob("optimization_log_*.csv"))

    if not log_files:
        raise FileNotFoundError(
            f"No optimization_log_*.csv file found inside: {RESULTS_DIR}\n"
            "Please run the optimization script first to generate a log."
        )

    return log_files[-1]


def normalize_column_name(name: str) -> str:
    """Remove surrounding spaces and compress internal whitespace."""
    return " ".join(str(name).strip().split())


def scale_normalized_to_real(df: pd.DataFrame) -> pd.DataFrame:
    """Convert normalized recipe values from 0-1 to physical recipe units."""
    df = df.copy()

    for var in RECIPE_COLUMNS:
        norm_col = f"{var}_norm"
        low, high = VARIABLE_RANGES[var]
        df[var] = low + df[norm_col] * (high - low)

    return df


def load_and_clean_log(log_file: Path) -> pd.DataFrame:
    """
    Load optimization log and clean column names.

    This script expects the log, where recipe variables
    are normalized and fidelity is stored as Fidelity_Index:
        0 -> micro, 1 -> bench, 2 -> pilot
    """
    df = pd.read_csv(log_file)

    # Clean names like ' T       ' -> 'T'
    df.columns = [normalize_column_name(c) for c in df.columns]

    # Rename log columns to analysis-friendly names
    rename_map = {
        "T": "T_norm",
        "pH_Wert": "pH_norm",
        "pH": "pH_norm",  # fallback
        "F1": "F1_norm",
        "F2": "F2_norm",
        "F3": "F3_norm",
        "Targetvalue_Y": "Y",
        "Step_Cost": "cost_eur",
        "Cost_Cummulated": "cumulative_cost_eur",
        "Cost_Cumulated": "cumulative_cost_eur",
        "Cost_Cumulative": "cumulative_cost_eur",
    }
    df = df.rename(columns=rename_map)

    required = [
        "Iteration", "T_norm", "pH_norm", "F1_norm", "F2_norm", "F3_norm",
        "Fidelity_Index", "Y", "cost_eur", "cumulative_cost_eur",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        print("Available columns after cleaning:")
        for col in df.columns:
            print(" -", col)
        raise ValueError(f"Missing required columns: {missing}")

    # Convert important columns to numeric
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows without measured Y or fidelity
    df = df.dropna(subset=["Iteration", "Fidelity_Index", "Y"])

    # Convert fidelity index to scale name
    df["Fidelity_Index"] = df["Fidelity_Index"].round().astype(int)
    df["scale"] = df["Fidelity_Index"].map(SCALE_MAP)

    # Convert normalized recipe variables to physical units
    df = scale_normalized_to_real(df)

    # Keep useful order, but preserve extra columns if needed
    useful = [
        "Iteration", "scale", "Fidelity_Index",
        "T_norm", "pH_norm", "F1_norm", "F2_norm", "F3_norm",
        "T", "pH", "F1", "F2", "F3",
        "Y", "cost_eur", "cumulative_cost_eur",
    ]
    existing_useful = [c for c in useful if c in df.columns]
    other_cols = [c for c in df.columns if c not in existing_useful]
    df = df[existing_useful + other_cols]

    return df


def select_initial_iteration(df: pd.DataFrame) -> pd.DataFrame:
    """Select only initial experiments."""
    initial_df = df[df["Iteration"] == 0].copy()

    if len(initial_df) == 0:
        raise ValueError(
            "No rows with Iteration == 0 were found. "
            "Cannot perform initial-iteration Task 1/2 analysis."
        )

    return initial_df


# Task 1: visualizations from initial data

def plot_y_vs_variables(df: pd.DataFrame) -> None:
    """Create Y-vs-variable scatter plots with separate scale labels."""
    labels = {
        "T": "Temperature T [°C]",
        "pH": "pH [-]",
        "F1": "Glucose feed F1 [g/L/h]",
        "F2": "Glutamine feed F2 [g/L/h]",
        "F3": "Rich-media feed F3 [g/L/h]",
    }

    for var in RECIPE_COLUMNS:
        plt.figure(figsize=(7, 5))

        for scale in ["micro", "bench", "pilot"]:
            subset = df[df["scale"] == scale]
            if len(subset) == 0:
                continue
            plt.scatter(subset[var], subset["Y"], label=scale, alpha=0.75)

        plt.xlabel(labels[var])
        plt.ylabel("Product amount Y")
        plt.title(f"Initial experiments: Y vs {var} at each scale")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        save_path = FIGURES_DIR / f"Initial_Y_vs_{var}.png"
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"Saved figure: {save_path}")


def plot_y_distribution_by_scale(df: pd.DataFrame) -> None:
    """Create a boxplot of Y by scale using initial experiments."""
    data = []
    labels = []

    for scale in ["micro", "bench", "pilot"]:
        values = df.loc[df["scale"] == scale, "Y"].dropna().values
        if len(values) > 0:
            data.append(values)
            labels.append(scale)

    if not data:
        print("No scale data available for Y distribution plot.")
        return

    plt.figure(figsize=(6, 5))
    plt.boxplot(data, labels=labels)
    plt.xlabel("Scale")
    plt.ylabel("Product amount Y")
    plt.title("Initial experiments: Y distribution by scale")
    plt.grid(True, axis="y")
    plt.tight_layout()
    save_path = FIGURES_DIR / "Initial_Y_distribution_by_scale.png"
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved figure: {save_path}")


def visualizations(df: pd.DataFrame) -> None:
    print("\nINITIAL EXPLORATORY VISUALIZATION")
    plot_y_vs_variables(df)
    plot_y_distribution_by_scale(df)


# Task 2: noise and bias from initial data

def noise_analysis(df: pd.DataFrame, rounding_digits: int = 3) -> pd.DataFrame:
    """
    Estimate measurement noise using repeated or near-repeated recipes.

    Noise can be estimated only when the same/near-same recipe appears more than
    once at the same scale. Since the initial design usually explores many
    different recipes, repeated points may be absent.
    """
    print("INITIAL NOISE ANALYSIS")

    work = df.copy()

    rounded_cols = []
    for col in RECIPE_COLUMNS:
        rounded_col = f"{col}_rounded"
        work[rounded_col] = work[col].round(rounding_digits)
        rounded_cols.append(rounded_col)

    noise_table = (
        work.groupby(["scale"] + rounded_cols)["Y"]
        .agg(count="count", mean_Y="mean", std_Y="std")
        .reset_index()
    )

    repeated_noise = noise_table[noise_table["count"] > 1].copy()
    repeated_noise.to_csv(TABLES_DIR / "initial_repeated_noise_table.csv", index=False)

    if len(repeated_noise) == 0:
        print("No repeated or near-repeated recipes found in the initial data.")
        print("Direct noise estimation is therefore not possible from the initial log.")
        return repeated_noise

    print("Repeated or near-repeated recipes found:")
    print(repeated_noise)

    scale_noise = (
        repeated_noise.groupby("scale")["std_Y"]
        .mean()
        .reset_index(name="mean_std_Y")
    )
    scale_noise.to_csv(TABLES_DIR / "initial_noise_per_scale.csv", index=False)

    plt.figure(figsize=(6, 4))
    plt.bar(scale_noise["scale"], scale_noise["mean_std_Y"])
    plt.xlabel("Scale")
    plt.ylabel("Mean standard deviation of Y")
    plt.title("Initial experiments: estimated noise per scale")
    plt.grid(True, axis="y")
    plt.tight_layout()
    save_path = FIGURES_DIR / "initial_noise_per_scale.png"
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved figure: {save_path}")

    return repeated_noise


def exact_bias_analysis(df: pd.DataFrame, rounding_digits: int = 3) -> pd.DataFrame:
    """
    Estimate exact/rounded bias from recipes tested at multiple scales.

    Bias definitions:
        bench_bias = bench Y - pilot Y
        micro_bias = micro Y - pilot Y

    This requires the same or rounded-same recipe to exist at both lower scale
    and pilot scale. If no common recipes exist, this function reports that
    exact bias cannot be estimated from the initial design.
    """
    print("\nINITIAL EXACT/ROUNDED BIAS ANALYSIS")

    work = df.copy()

    rounded_cols = []
    for col in RECIPE_COLUMNS:
        rounded_col = f"{col}_rounded"
        work[rounded_col] = work[col].round(rounding_digits)
        rounded_cols.append(rounded_col)

    avg = (
        work.groupby(rounded_cols + ["scale"])["Y"]
        .mean()
        .reset_index()
    )

    comparison = avg.pivot_table(index=rounded_cols, columns="scale", values="Y").reset_index()
    comparison.to_csv(TABLES_DIR / "initial_scale_comparison_rounded.csv", index=False)

    if "pilot" not in comparison.columns:
        print("No Pilot data found in initial data. Bias relative to Pilot cannot be estimated.")
        return comparison

    if "bench" in comparison.columns:
        comparison["bench_bias"] = comparison["bench"] - comparison["pilot"]
        bench_pilot = comparison[rounded_cols + ["bench", "pilot", "bench_bias"]].dropna()
        bench_pilot.to_csv(TABLES_DIR / "initial_bench_pilot_bias.csv", index=False)
        if len(bench_pilot) > 0:
            print("Bench-Pilot bias from common rounded initial recipes:")
            print(bench_pilot)
            print("Average Bench bias:", bench_pilot["bench_bias"].mean())
        else:
            print("No common rounded Bench-Pilot recipes found in initial data.")

    if "micro" in comparison.columns:
        comparison["micro_bias"] = comparison["micro"] - comparison["pilot"]
        micro_pilot = comparison[rounded_cols + ["micro", "pilot", "micro_bias"]].dropna()
        micro_pilot.to_csv(TABLES_DIR / "initial_micro_pilot_bias.csv", index=False)
        if len(micro_pilot) > 0:
            print("Micro-Pilot bias from common rounded initial recipes:")
            print(micro_pilot)
            print("Average Micro bias:", micro_pilot["micro_bias"].mean())
        else:
            print("No common rounded Micro-Pilot recipes found in initial data.")

    return comparison


def nearest_pilot_bias(df: pd.DataFrame) -> pd.DataFrame:
    """
    Approximate fallback bias estimate for initial data.

    For each micro/bench initial experiment, find the nearest Pilot recipe in
    normalized recipe space and compute:
        approximate bias = lower-scale Y - nearest Pilot Y

    This is only an approximation, because the recipes are similar but not
    identical. Use it only to discuss scale behavior, not as an exact bias.
    """
    print("\nINITIAL APPROXIMATE NEAREST-PILOT BIAS")

    pilot = df[df["scale"] == "pilot"].copy()
    lower = df[df["scale"].isin(["micro", "bench"])].copy()

    if len(pilot) == 0 or len(lower) == 0:
        print("Nearest-pilot bias cannot be calculated because pilot or lower-scale initial data is missing.")
        return pd.DataFrame()

    rows = []

    pilot_x = pilot[NORMALIZED_COLUMNS].to_numpy(dtype=float)
    pilot_y = pilot["Y"].to_numpy(dtype=float)

    for _, row in lower.iterrows():
        x = row[NORMALIZED_COLUMNS].to_numpy(dtype=float)
        distances = np.linalg.norm(pilot_x - x, axis=1)
        nearest_index = int(np.argmin(distances))

        nearest_pilot_row = pilot.iloc[nearest_index]

        rows.append({
            "scale": row["scale"],
            "Y_scale": row["Y"],
            "nearest_pilot_Y": pilot_y[nearest_index],
            "approx_bias": row["Y"] - pilot_y[nearest_index],
            "distance_to_nearest_pilot_norm_space": distances[nearest_index],
            "T": row["T"],
            "pH": row["pH"],
            "F1": row["F1"],
            "F2": row["F2"],
            "F3": row["F3"],
            "nearest_pilot_T": nearest_pilot_row["T"],
            "nearest_pilot_pH": nearest_pilot_row["pH"],
            "nearest_pilot_F1": nearest_pilot_row["F1"],
            "nearest_pilot_F2": nearest_pilot_row["F2"],
            "nearest_pilot_F3": nearest_pilot_row["F3"],
        })

    result = pd.DataFrame(rows)
    result.to_csv(TABLES_DIR / "initial_nearest_pilot_bias.csv", index=False)

    summary = (
        result.groupby("scale")["approx_bias"]
        .agg(count="count", mean_bias="mean", std_bias="std")
        .reset_index()
    )
    summary.to_csv(TABLES_DIR / "initial_nearest_pilot_bias_summary.csv", index=False)

    print("Approximate nearest-pilot bias summary from initial data:")
    print(summary)

    plt.figure(figsize=(6, 4))
    plt.bar(summary["scale"], summary["mean_bias"])
    plt.xlabel("Scale")
    plt.ylabel("Approximate bias: scale Y - nearest Pilot Y")
    plt.title("Initial experiments: approximate bias relative to nearest Pilot")
    plt.grid(True, axis="y")
    plt.tight_layout()
    save_path = FIGURES_DIR / "initial_nearest_pilot_bias_summary.png"
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved figure: {save_path}")

    return result


def analysis(df: pd.DataFrame) -> None:
    print("\nINITIAL SCALE NOISE AND BIAS ANALYSIS")
    noise_analysis(df)
    exact_bias_analysis(df)
    nearest_pilot_bias(df)


# Summary tables for initial data

def save_summary_tables(df: pd.DataFrame) -> None:
    """Save basic scale-wise summary tables for initial experiments."""
    scale_summary = (
        df.groupby("scale")
        .agg(
            count=("Y", "count"),
            mean_Y=("Y", "mean"),
            std_Y=("Y", "std"),
            min_Y=("Y", "min"),
            max_Y=("Y", "max"),
            total_cost_eur=("cost_eur", "sum"),
        )
        .reset_index()
    )
    scale_summary.to_csv(TABLES_DIR / "initial_scale_summary.csv", index=False)

    print("\nInitial scale summary:")
    print(scale_summary)


# Main

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task 1 and Task 2 initial-experiment analysis for MLME bioreactor logs."
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional path to a specific optimization log CSV. If not given, latest results/optimization_log_*.csv is used.",
    )
    args = parser.parse_args()

    if args.log_file is None:
        log_file = find_latest_log_file()
    else:
        log_file = Path(args.log_file)
        if not log_file.is_absolute():
            log_file = PROJECT_DIR / log_file

    print(f"Reading log file: {log_file}")
    full_df = load_and_clean_log(log_file)

    cleaned_full_path = TABLES_DIR / "cleaned_full_optimization_log_reference.csv"
    full_df.to_csv(cleaned_full_path, index=False)
    print(f"Saved cleaned full log for reference: {cleaned_full_path}")

    initial_df = select_initial_iteration(full_df)

    cleaned_initial_path = TABLES_DIR / "cleaned_initial_iteration_log.csv"
    initial_df.to_csv(cleaned_initial_path, index=False)
    print(f"Saved cleaned initial log: {cleaned_initial_path}")

    print("\nInitial dataframe preview:")
    print(initial_df.head())

    print("\nNumber of initial experiments:", len(initial_df))
    print("Initial experiments by scale:")
    print(initial_df["scale"].value_counts())

    save_summary_tables(initial_df)
    visualizations(initial_df)
    analysis(initial_df)

    print("\nDone.")
    print(f"Figures saved in: {FIGURES_DIR}")
    print(f"Tables saved in:  {TABLES_DIR}")


if __name__ == "__main__":
    main()