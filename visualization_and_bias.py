from pathlib import Path
from typing import Optional
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Paths

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = PROJECT_DIR / "results"

# Outputs
FIGURES_DIR = DEFAULT_RESULTS_DIR / "figures"
TABLES_DIR = DEFAULT_RESULTS_DIR / "tables"

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

def normalize_column_name(name: str) -> str:
    """Remove leading/trailing spaces and compress internal whitespace."""
    return " ".join(str(name).strip().split())


def resolve_input_file(user_path: Optional[str], candidate_names: list[str]) -> Path:
    """
    Resolve input file path.

    Search order:
    1. user-provided path, if given
    2. results/<candidate>
    3. project root/<candidate>
    """
    if user_path is not None:
        path = Path(user_path)
        if not path.is_absolute():
            path = PROJECT_DIR / path
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return path

    for name in candidate_names:
        candidate = DEFAULT_RESULTS_DIR / name
        if candidate.exists():
            return candidate

    for name in candidate_names:
        candidate = PROJECT_DIR / name
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find any of these files:\n"
        + "\n".join(f" - {name}" for name in candidate_names)
        + f"\nSearched in:\n - {DEFAULT_RESULTS_DIR}\n - {PROJECT_DIR}"
    )


def scale_normalized_to_real(df: pd.DataFrame) -> pd.DataFrame:
    """Convert normalized recipe variables from [0,1] to physical units."""
    df = df.copy()

    for var in RECIPE_COLUMNS:
        norm_col = f"{var}_norm"
        low, high = VARIABLE_RANGES[var]
        df[var] = low + df[norm_col] * (high - low)

    return df


def load_and_clean_csv(path: Path, forced_scale: Optional[str] = None) -> pd.DataFrame:
    """
    Load one CSV and clean it into a common format.

    Expected input columns from the simulator/BO logs:
        T, pH_Wert, F1, F2, F3, Fidelity_Index, Targetvalue_Y,
        Step_Cost, Cost_Cummulated

    Output columns:
        scale, T_norm, pH_norm, F1_norm, F2_norm, F3_norm,
        T, pH, F1, F2, F3, Y, cost_eur, cumulative_cost_eur
    """
    df = pd.read_csv(path)
    df.columns = [normalize_column_name(c) for c in df.columns]

    rename_map = {
        "T": "T_norm",
        "pH_Wert": "pH_norm",
        "pH": "pH_norm",
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
        "T_norm", "pH_norm", "F1_norm", "F2_norm", "F3_norm",
        "Fidelity_Index", "Y", "cost_eur", "cumulative_cost_eur",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"\nAvailable columns in {path.name}:")
        for col in df.columns:
            print(" -", col)
        raise ValueError(f"Missing required columns in {path.name}: {missing}")

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Iteration" in df.columns:
        df["Iteration"] = pd.to_numeric(df["Iteration"], errors="coerce")

    df = df.dropna(subset=[
        "T_norm", "pH_norm", "F1_norm", "F2_norm", "F3_norm",
        "Fidelity_Index", "Y",
    ]).copy()

    df["Fidelity_Index"] = df["Fidelity_Index"].round().astype(int)
    df["scale"] = df["Fidelity_Index"].map(SCALE_MAP)

    if forced_scale is not None:
        df["scale"] = forced_scale
        reverse_map = {"micro": 0, "bench": 1, "pilot": 2}
        df["Fidelity_Index"] = reverse_map[forced_scale]

    df = scale_normalized_to_real(df)

    useful = [
        "Iteration", "scale", "Fidelity_Index",
        "T_norm", "pH_norm", "F1_norm", "F2_norm", "F3_norm",
        "T", "pH", "F1", "F2", "F3",
        "Y", "cost_eur", "cumulative_cost_eur",
    ]
    useful_existing = [c for c in useful if c in df.columns]
    other_cols = [c for c in df.columns if c not in useful_existing]
    return df[useful_existing + other_cols]


def add_recipe_id(df: pd.DataFrame, rounding_digits: int = 8) -> pd.DataFrame:
    """
    Add recipe_id by grouping the normalized recipe coordinates.

    The noise/bias files contain 16 recipes repeated 10 times.
    Since recipes are continuous, grouping is done after rounding.
    """
    df = df.copy()

    rounded_cols = []
    for col in NORMALIZED_COLUMNS:
        rounded_col = f"{col}_rounded"
        df[rounded_col] = df[col].round(rounding_digits)
        rounded_cols.append(rounded_col)

    # group ids are assigned based on unique rounded normalized recipes
    df["recipe_id"] = df.groupby(rounded_cols, sort=True).ngroup() + 1

    return df


# Task 1: Y dependence visualizations

def summarize_visualisation_data(vis_df: pd.DataFrame) -> pd.DataFrame:
    """Create scale-wise summary for visualisation.csv."""
    summary = (
        vis_df.groupby("scale")
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

    summary.to_csv(TABLES_DIR / "visualisation_scale_summary.csv", index=False)
    return summary


def plot_y_vs_variables(vis_df: pd.DataFrame) -> None:
    """Plot product amount Y against each recipe variable for all scales."""
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
            subset = vis_df[vis_df["scale"] == scale]
            if len(subset) == 0:
                continue
            plt.scatter(subset[var], subset["Y"], label=scale, alpha=0.75)

        plt.xlabel(labels[var])
        plt.ylabel("Product amount Y")
        plt.title(f"Y dependence on {var}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        save_path = FIGURES_DIR / f"Y_vs_{var}.png"
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"Saved figure: {save_path}")


def plot_y_distribution_by_scale(vis_df: pd.DataFrame) -> None:
    """Plot Y distribution by scale for visualisation.csv."""
    data = []
    labels = []

    for scale in ["micro", "bench", "pilot"]:
        values = vis_df.loc[vis_df["scale"] == scale, "Y"].dropna().values
        if len(values) > 0:
            data.append(values)
            labels.append(scale)

    if not data:
        print("No data available for Y distribution plot.")
        return

    plt.figure(figsize=(6, 5))
    plt.boxplot(data, labels=labels)
    plt.xlabel("Scale")
    plt.ylabel("Product amount Y")
    plt.title("Y distribution by scale")
    plt.grid(True, axis="y")
    plt.tight_layout()

    save_path = FIGURES_DIR / "Y_distribution_by_scale.png"
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved figure: {save_path}")


def visualisation_analysis(vis_df: pd.DataFrame) -> None:
    print("\nY DEPENDENCE VISUALIZATION")

    summary = summarize_visualisation_data(vis_df)
    print("\nscale summary from visualisation.csv:")
    print(summary)

    vis_df.to_csv(TABLES_DIR / "cleaned_visualisation_data.csv", index=False)

    plot_y_vs_variables(vis_df)
    plot_y_distribution_by_scale(vis_df)


# Task 2: Noise and bias analysis

def load_noise_bias_data(micro_path: Path, bench_path: Path, pilot_path: Path) -> pd.DataFrame:
    """Load and combine the three repeated-recipe noise/bias datasets."""
    micro = load_and_clean_csv(micro_path, forced_scale="micro")
    bench = load_and_clean_csv(bench_path, forced_scale="bench")
    pilot = load_and_clean_csv(pilot_path, forced_scale="pilot")

    combined = pd.concat([micro, bench, pilot], ignore_index=True)
    combined = add_recipe_id(combined, rounding_digits=8)

    combined.to_csv(TABLES_DIR / "cleaned_combined_noise_bias_data.csv", index=False)

    return combined


def recipe_noise_analysis(nb_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Estimate measurement noise.

    For each scale and repeated recipe:
        mean_Y = average product amount
        std_Y  = standard deviation across 10 repeats
        cv_Y   = std_Y / |mean_Y|

    Then summarize noise per scale.
    """
    recipe_noise = (
        nb_df.groupby(["scale", "recipe_id"])
        .agg(
            repeat_count=("Y", "count"),
            mean_Y=("Y", "mean"),
            std_Y=("Y", "std"),
            min_Y=("Y", "min"),
            max_Y=("Y", "max"),
            T=("T", "mean"),
            pH=("pH", "mean"),
            F1=("F1", "mean"),
            F2=("F2", "mean"),
            F3=("F3", "mean"),
        )
        .reset_index()
    )

    recipe_noise["cv_Y"] = recipe_noise["std_Y"] / recipe_noise["mean_Y"].abs()

    scale_noise = (
        recipe_noise.groupby("scale")
        .agg(
            recipes=("recipe_id", "count"),
            mean_repeat_count=("repeat_count", "mean"),
            mean_Y=("mean_Y", "mean"),
            mean_std_Y=("std_Y", "mean"),
            median_std_Y=("std_Y", "median"),
            mean_cv_Y=("cv_Y", "mean"),
            median_cv_Y=("cv_Y", "median"),
        )
        .reset_index()
    )

    recipe_noise.to_csv(TABLES_DIR / "recipe_noise_table.csv", index=False)
    scale_noise.to_csv(TABLES_DIR / "noise_per_scale.csv", index=False)

    print("\nnoise per scale:")
    print(scale_noise)

    plt.figure(figsize=(6, 4))
    plt.bar(scale_noise["scale"], scale_noise["mean_std_Y"])
    plt.xlabel("Scale")
    plt.ylabel("Mean standard deviation of Y")
    plt.title("Measurement noise per scale")
    plt.grid(True, axis="y")
    plt.tight_layout()

    save_path = FIGURES_DIR / "noise_per_scale.png"
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved figure: {save_path}")

    plt.figure(figsize=(6, 4))
    plt.bar(scale_noise["scale"], scale_noise["mean_cv_Y"])
    plt.xlabel("Scale")
    plt.ylabel("Mean coefficient of variation")
    plt.title("Relative noise per scale")
    plt.grid(True, axis="y")
    plt.tight_layout()

    save_path = FIGURES_DIR / "relative_noise_per_scale.png"
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved figure: {save_path}")

    return recipe_noise, scale_noise


def bias_analysis(recipe_noise: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Estimate scale bias using recipe-wise mean Y values.

    Since the same 16 recipes were repeated at all three scales:
        micro_bias = micro_mean_Y - pilot_mean_Y
        bench_bias = bench_mean_Y - pilot_mean_Y

    Positive bias: lower scale overestimates Pilot.
    Negative bias: lower scale underestimates Pilot.
    """
    mean_by_recipe_scale = recipe_noise[["recipe_id", "scale", "mean_Y", "std_Y", "repeat_count"]].copy()

    pivot = mean_by_recipe_scale.pivot(index="recipe_id", columns="scale", values="mean_Y").reset_index()
    pivot.to_csv(TABLES_DIR / "mean_Y_and_scale.csv", index=False)

    if "pilot" not in pivot.columns:
        raise ValueError("Pilot scale is missing. Bias relative to Pilot cannot be calculated.")

    if "micro" in pivot.columns:
        pivot["micro_bias"] = pivot["micro"] - pivot["pilot"]

    if "bench" in pivot.columns:
        pivot["bench_bias"] = pivot["bench"] - pivot["pilot"]

    bias_cols = ["recipe_id"]
    for c in ["micro", "bench", "pilot", "micro_bias", "bench_bias"]:
        if c in pivot.columns:
            bias_cols.append(c)

    bias_table = pivot[bias_cols].copy()
    bias_table.to_csv(TABLES_DIR / "bias_relative_to_pilot.csv", index=False)

    summary_rows = []
    if "micro_bias" in bias_table.columns:
        summary_rows.append({
            "scale": "micro",
            "mean_bias": bias_table["micro_bias"].mean(),
            "median_bias": bias_table["micro_bias"].median(),
            "std_bias": bias_table["micro_bias"].std(),
            "min_bias": bias_table["micro_bias"].min(),
            "max_bias": bias_table["micro_bias"].max(),
        })

    if "bench_bias" in bias_table.columns:
        summary_rows.append({
            "scale": "bench",
            "mean_bias": bias_table["bench_bias"].mean(),
            "median_bias": bias_table["bench_bias"].median(),
            "std_bias": bias_table["bench_bias"].std(),
            "min_bias": bias_table["bench_bias"].min(),
            "max_bias": bias_table["bench_bias"].max(),
        })

    bias_summary = pd.DataFrame(summary_rows)
    bias_summary.to_csv(TABLES_DIR / "bias_summary_relative_to_pilot.csv", index=False)

    print("\nBias summary relative to Pilot:")
    print(bias_summary)

    plt.figure(figsize=(6, 4))
    plt.bar(bias_summary["scale"], bias_summary["mean_bias"])
    plt.axhline(0, linewidth=1)
    plt.xlabel("Lower-fidelity scale")
    plt.ylabel("Mean bias: scale Y - Pilot Y")
    plt.title("Mean bias relative to Pilot")
    plt.grid(True, axis="y")
    plt.tight_layout()

    save_path = FIGURES_DIR / "mean_bias_relative_to_pilot.png"
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved figure: {save_path}")

    # Recipe-wise comparison plot
    plt.figure(figsize=(8, 5))
    for scale in ["micro", "bench", "pilot"]:
        if scale in pivot.columns:
            plt.plot(pivot["recipe_id"], pivot[scale], marker="o", label=scale)

    plt.xlabel("Recipe ID")
    plt.ylabel("Mean product amount Y")
    plt.title("Mean Y for repeated recipes across scales")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = FIGURES_DIR / "mean_Y_by_recipe_across_scales.png"
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved figure: {save_path}")

    return bias_table, bias_summary



def noise_bias_analysis(nb_df: pd.DataFrame) -> None:
    print("\nNOISE AND BIAS ANALYSIS")

    recipe_noise, scale_noise = recipe_noise_analysis(nb_df)
    bias_table, bias_summary = bias_analysis(recipe_noise)



# Main

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MLME Task 1 and Task 2 analysis using separate visualisation and repeated noise/bias CSV files."
    )

    parser.add_argument(
        "--visualisation-file",
        type=str,
        default=None,
        help="Path to visualisation.csv. If not given, script searches in results/ and project root.",
    )

    parser.add_argument(
        "--micro-file",
        type=str,
        default=None,
        help="Path to Microplate_noise_bias_analysis.csv.",
    )

    parser.add_argument(
        "--bench-file",
        type=str,
        default=None,
        help="Path to Bench_noise_bias_analysis.csv.",
    )

    parser.add_argument(
        "--pilot-file",
        type=str,
        default=None,
        help="Path to pilot_noise_bias_analysis.csv.",
    )

    args = parser.parse_args()

    visualisation_file = resolve_input_file(
        args.visualisation_file,
        ["visualisation.csv", "visualization.csv"]
    )
    micro_file = resolve_input_file(
        args.micro_file,
        ["Microplate_noise_bias_analysis.csv", "microplate_noise_bias_analysis.csv"]
    )
    bench_file = resolve_input_file(
        args.bench_file,
        ["Bench_noise_bias_analysis.csv", "bench_noise_bias_analysis.csv"]
    )
    pilot_file = resolve_input_file(
        args.pilot_file,
        ["pilot_noise_bias_analysis.csv", "Pilot_noise_bias_analysis.csv"]
    )

    print("Using input files:")
    print(f" - Task 1 visualisation: {visualisation_file}")
    print(f" - Task 2 microplate:    {micro_file}")
    print(f" - Task 2 bench:         {bench_file}")
    print(f" - Task 2 pilot:         {pilot_file}")

    # Task 1
    vis_df = load_and_clean_csv(visualisation_file)
    visualisation_analysis(vis_df)

    # Task 2
    nb_df = load_noise_bias_data(
        micro_path=micro_file,
        bench_path=bench_file,
        pilot_path=pilot_file,
    )
    noise_bias_analysis(nb_df)

    print("\nDone.")
    print(f"Figures saved in: {FIGURES_DIR}")
    print(f"Tables saved in:  {TABLES_DIR}")


if __name__ == "__main__":
    main()