"""Run global sensitivity analysis for the optimization model."""

import os
import torch
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_mll
from botorch.models.deterministic import DeterministicModel
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition import PosteriorMean
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.optim.optimize import optimize_acqf_mixed
import pandas as pd
import numpy as np
from basic_client import BioreactorClient
import time
from scipy.stats import qmc
import numpy as np
from datetime import datetime
from SALib.sample import saltelli
from SALib.analyze import sobol
from mfbo_v1 import FidelityScaler, problem

def run_global_sensitivity_analysis(model, scaler):
    """Run Sobol sensitivity analysis on the surrogate model."""
    print("\n--- Start global sensitivity analysis on pilot level ---")
    
    # Define the sensitivity-analysis problem.
    problem_def = {
        'num_vars': 5,
        'names': ['T', 'pH', 'F1', 'F2', 'F3'],
        'bounds': [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
    }
    
    # Generate Saltelli samples.
    X_samples = saltelli.sample(problem_def, 1024)
    
    # Prepare the inputs as tensors.
    X_tensor = torch.tensor(X_samples, **tkwargs)
    
    # Fix fidelity at pilot level.
    fidelity_col = torch.ones((X_tensor.shape[0], 1), **tkwargs)
    X_full = torch.cat([X_tensor, fidelity_col], dim=1)
    
    # Predict with the surrogate model.
    model.eval()
    with torch.no_grad():
        posterior = model.posterior(X_full)
        Y_pred_norm = posterior.mean.squeeze(-1) 
        
    # Convert predictions back to the real scale.
    Y_pred = scaler.denormalize(Y_pred_norm, fidelity=1.0).cpu().numpy()
    
    # Run Sobol analysis.
    Si = sobol.analyze(problem_def, Y_pred, print_to_console=False)
    
    # Print the sensitivity results.
    print(f"{'Variable':<10} | {'S1 (Haupteffekt)':<20} | {'ST (Gesamteffekt)':<20}")
    print("-" * 55)
    for i, name in enumerate(problem_def['names']):
        s1 = Si['S1'][i]
        st = Si['ST'][i]
        print(f"{name:<10} | {s1:<20.4f} | {st:<20.4f}")
        
    return Si

def run_real_pilot_gsa(client):
    """Run Sobol analysis with real pilot API evaluations."""
    print("\n--- Start global sensitivity analysis with real pilot runs ---")
    
    # Define the real-parameter bounds.
    problem_def = {
        'num_vars': 5,
        'names': ['T', 'pH', 'F1', 'F2', 'F3'],
        'bounds': [
            [20.0, 60.0],  # T
            [3.0, 9.5],    # pH
            [0.0, 2.0],    # F1
            [0.0, 2.0],    # F2
            [0.0, 2.0]     # F3
        ]
    }
    
    # Generate Saltelli samples.
    X_samples = saltelli.sample(problem_def, 16) 
    
    # Store the real outcomes.
    Y_results = np.zeros(X_samples.shape[0])
    total_cost_gsa = 0.0
    
    # Evaluate each point with the API at pilot fidelity.
    for i, row in enumerate(X_samples):
        result = client.run("pilot", row[0], row[1], row[2], row[3], row[4])
        
        Y_results[i] = result['Y']
        total_cost_gsa += result['cost_eur']
        
        print(f"Run {i+1}/{len(X_samples)} finished. Current GSA cost: {total_cost_gsa} EUR")
        
    # Run Sobol analysis on the real outcomes.
    Si = sobol.analyze(problem_def, Y_results, print_to_console=False)
    
    print("\n--- GSA Ergebnisse ---")
    print(f"{'Variable':<10} | {'S1 (Haupteffekt)':<20} | {'ST (Gesamteffekt)':<20}")
    print("-" * 55)
    for i, name in enumerate(problem_def['names']):
        print(f"{name:<10} | {Si['S1'][i]:<20.4f} | {Si['ST'][i]:<20.4f}")
        
    return Si

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
    
    return model, df, scaler





if __name__ == "__main__":
    # Set the current optimization log file.
    DATEINAME = r"results\optimization_log_2026-07-11_09-43-48.csv"
    model, daten_df, scaler = evaluate_final_model(DATEINAME)
    # run_global_sensitivity_analysis(model, scaler)
    run_real_pilot_gsa(client)
