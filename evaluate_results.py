"""Evaluate and visualize the optimization results."""

import pandas as pd
import torch
import matplotlib.pyplot as plt
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_mll
from botorch.acquisition import PosteriorMean
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.optim.optimize import optimize_acqf
from mfbo_v1 import problem, FidelityScaler
from basic_client import BioreactorClient

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

bounds = torch.tensor(
    [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], 
    **tkwargs
)

USER = "group14"
PASSWORD = "ip00y-p92nc-9smqo"
BASE_URL = "https://mlme26biosim.org"

VARIABLE_RANGES = {
    "T": (20.0, 60.0),
    "pH": (3.0, 9.5),
    "F1": (0.0, 2.0),
    "F2": (0.0, 2.0),
    "F3": (0.0, 2.0),
}

SCALES = ["micro", "bench", "pilot"]

client = BioreactorClient(BASE_URL)
client.login(USER, PASSWORD)

def evaluate_final_model(csv_path):
    """Fit the final model and return the best recommendation."""
    print("Load data from csv file...")
    df = pd.DataFrame(pd.read_csv(csv_path))
    scaler = FidelityScaler()
    feature_cols = ["T", "pH_Wert", "F1", "F2", "F3", "Fidelity_Scaled"]
    train_x = torch.tensor(df[feature_cols].values, **tkwargs)
    train_obj = torch.tensor(df["Targetvalue_Y"].values, **tkwargs).unsqueeze(-1)
    train_obj_norm = scaler.transform(train_x, train_obj)
    print("Train final model with all data...")
    # Fit the surrogate model on all available data.
    model = SingleTaskMultiFidelityGP(
        train_x, train_obj_norm, data_fidelities=[5]
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    
    rec_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model),
        d=6, columns=[5], values=[1.0],
    )
    
    final_rec_x, final_rec_y = optimize_acqf(
        acq_function=rec_acqf,
        bounds=bounds[:, :-1],
        q=1, num_restarts=10, raw_samples=512,
    )
    
    final_rec_x_full = rec_acqf._construct_X_full(final_rec_x)
    final_rec_y_real, _ = problem(final_rec_x_full)

    best_y_val = final_rec_y.item()
    best_y_val_denorm = scaler.denormalize(best_y_val, fidelity=1.0).item()
    best_y_val_real = final_rec_y_real.item()

    mins = torch.tensor([r[0] for r in VARIABLE_RANGES.values()], **tkwargs)
    maxs = torch.tensor([r[1] for r in VARIABLE_RANGES.values()], **tkwargs)

    final_rec_x_denormal = mins + final_rec_x[0:5] * (maxs - mins)

    print(f"\n=== FINAL MODEL-ANALYSis ===")
    print(f"Best theorticall value (Pilot): {best_y_val_denorm:.3f}")
    print(f"Best real value (Pilot):        {best_y_val_real:.3f}")
    print(f"Optimal recipe (denormalised):     {final_rec_x_denormal.squeeze().tolist()}")
    
    return model, df

def plot_fast_convergence(df):
    """Compare model predictions with real outcomes over time."""
    print("\nStart evaluating previous results...")
    # Keep only pilot observations for the running maximum.
    df['Pilot_Y_Only'] = df['Targetvalue_Y'].where(df['Fidelity_Index'] == 2)
    
    df['Running_Max_Y'] = df['Pilot_Y_Only'].cummax().ffill()
    
    df['Running_Max_Y'] = df['Running_Max_Y'].bfill()
    
    df_opt = df[df['Iteration'] > 0].copy()
    df_opt = df_opt.iloc[:-1]
    cumulative_costs = df_opt["Cost_Cummulated"].values
    predicted_max_y = df_opt["Pred_Max_Y"].values
    real_max_y = df_opt["Running_Max_Y"].values
    iteration = df_opt["Iteration"].values
    
    true_recommendation_y = []
    
    for index, row in df_opt.iterrows():
        x_tensor = torch.tensor([[
            row["Pred_Opt_T"], 
            row["Pred_Opt_pH"], 
            row["Pred_Opt_F1"], 
            row["Pred_Opt_F2"], 
            row["Pred_Opt_F3"], 
            1.0
        ]], **tkwargs)
        
        y_real, _ = problem(x_tensor)
        true_recommendation_y.append(y_real.item())
        
        print(f"Iter {int(row['Iteration'])} | model predicts: {row['Pred_Max_Y']:.2f} | Real value: {y_real.item():.2f}")

    plt.figure(figsize=(10, 6))

    ax1 = plt.gca()
    
    ax1.plot(iteration, predicted_max_y, label='Model prediction (Theoretical Maximum)', 
             color='blue', linewidth=2, linestyle='-')
             
    ax1.plot(iteration, true_recommendation_y, label='Real value at predicted maximum', 
             color='red', linewidth=2, linestyle='-')
    
    ax1.step(iteration, real_max_y, label='actual previous measured maximum', 
             color='black', linewidth=2, linestyle='--', where='post')

    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Target value $Y$', fontsize=12)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(iteration, cumulative_costs, label='Cummulative cost', 
             color='purple', linewidth=2, linestyle='-')
    ax2.set_ylabel('Cummulative cost', fontsize=12)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    plt.title('Model-convergence', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_exploration_and_yield(csv_path="results/optimization_log.csv"):
    """Create a two-row dashboard for exploration and yield trends."""
    print("Lese CSV-Datei für die erweiterte Exploration ein...")
    df = pd.read_csv(csv_path)
    
    params = ["T", "pH_Wert", "F1", "F2", "F3"]
    
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.5, 1], hspace=0.3)
    
    axes_params = [fig.add_subplot(gs[0, i]) for i in range(5)]
    ax_yield = fig.add_subplot(gs[1, :])
    
    fig.suptitle('Optimierungs-Dashboard: Suchverhalten und relativer Ertrag', fontsize=18, fontweight='bold')
    
    colors = {0: 'lightblue', 1: 'orange', 2: 'red'}
    labels = {0: 'Micro (f=0)', 1: 'Bench (f=0.5)', 2: 'Pilot (f=1.0)'}
    
    # Scale bubble size per fidelity level.
    df['BubbleSize'] = 50.0  # Fallback-Standardgröße
    
    for fid in [0, 1, 2]:
        mask = df['Fidelity_Index'] == fid
        if mask.sum() > 0:
            y_vals = df.loc[mask, 'Targetvalue_Y'].fillna(0)
            y_min, y_max = y_vals.min(), y_vals.max()
            
            if y_max > y_min:
                # Skaliert die Punkte NUR innerhalb dieser Fidelity von 20 bis 250
                df.loc[mask, 'BubbleSize'] = 20 + 230 * ((y_vals - y_min) / (y_max - y_min))
            else:
                df.loc[mask, 'BubbleSize'] = 50.0

    # Top row: parameter traces over iterations.
    for i, param in enumerate(params):
        ax = axes_params[i]
        
        for fid in [0, 1, 2]:
            mask = df['Fidelity_Index'] == fid
            if mask.sum() > 0:
                ax.scatter(
                    df.loc[mask, 'Iteration'], 
                    df.loc[mask, param],
                    c=colors[fid], 
                    alpha=0.6, 
                    edgecolors='black', 
                    linewidth=0.5,
                    s=df.loc[mask, 'BubbleSize']  # Nimmt jetzt die relativierten Werte
                )
                
        ax.set_title(param, fontsize=14)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        if i == 0:
            ax.set_ylabel('Parameterwert (0-1)', fontsize=12)
            
        # Finales Optimum einzeichnen
        col_name = f"Pred_Opt_{param}" if f"Pred_Opt_{param}" in df.columns else f"Pred_Opt_{param.split('_')[0]}"
        if col_name in df.columns and pd.notna(df[col_name].iloc[-1]):
            ax.axhline(df[col_name].iloc[-1], color='green', linestyle=':', alpha=0.8)
        # Plot per-iteration predicted optimum for this parameter (if available)
        if col_name in df.columns:
            pred_param = pd.to_numeric(df[col_name], errors='coerce')
            if pred_param.notna().any():
                label_pred = 'Model predicted optimum' if i == 0 else None
                ax.plot(df['Iteration'].values, pred_param.values,
                        color='green', linewidth=1.5, linestyle='-', alpha=0.9,
                        label=label_pred)

    # Bottom row: measured yield over time.
    for fid in [0, 1, 2]:
        mask = df['Fidelity_Index'] == fid
        if mask.sum() > 0:
            ax_yield.scatter(
                df.loc[mask, 'Iteration'], 
                df.loc[mask, 'Targetvalue_Y'],
                c=colors[fid], 
                label=labels[fid],
                alpha=0.8, 
                edgecolors='black', 
                s=60
            )
            ax_yield.plot(df.loc[mask, 'Iteration'], df.loc[mask, 'Targetvalue_Y'], c=colors[fid], alpha=0.2)

    ax_yield.set_title('Gemessener Ertrag (Targetvalue_Y) über die Iterationen', fontsize=14)
    ax_yield.set_xlabel('Iteration', fontsize=12)
    ax_yield.set_ylabel('Ertrag (Y)', fontsize=12)
    ax_yield.grid(True, linestyle='--', alpha=0.5)
    
    ax_yield.legend(loc='upper left', fontsize=12)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Set the current optimization log file.
    DATEINAME = r"results\optimization_log_2026-07-11_06-38-26.csv"
    plot_exploration_and_yield(DATEINAME)
    model, daten_df = evaluate_final_model(DATEINAME)
    
    plot_fast_convergence(daten_df)