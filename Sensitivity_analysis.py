"""Run global sensitivity analysis for the optimization model.

SUPERSEDED by ``task5_sensitivity.py`` -- use that instead.

This file is kept only because ``run_real_pilot_gsa`` (a GSA against the real
API, rather than the surrogate) is an idea worth keeping. Two things about it
had to be disarmed first; see ``task5_gsa_postmortem.md`` for the full write-up.

  1. COST. ``saltelli.sample(problem, N)`` returns N*(2D+2) rows, NOT N. At the
     N=16 this file used, that is 192 real PILOT runs = 384,000 EUR, against a
     15,000 EUR benchmark budget -- and it was the live call in __main__. It now
     refuses to run without an explicit cost acknowledgement.
  2. The pilot-level surrogate GSA this file performs is the analysis that
     produced our wrong "T is inert" conclusion. ``task5_sensitivity.py`` now
     runs it at every fidelity with a design-coverage gate.

Note: importing this module imports ``mfbo_v1``, which logs into the API at
import time. ``problem()`` likewise performs a real (billed) API call.
"""

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

PILOT_COST_EUR = 2000.0


def run_real_pilot_gsa(client, n_base=16, budget_eur=0.0):
    """Run Sobol analysis with real pilot API evaluations.

    Saltelli expands n_base to n_base*(2D+2) runs -- at D=5 that is 12x the
    number you typed. Every row is a billed pilot run, so the sample count is
    computed and checked against ``budget_eur`` BEFORE anything is spent.
    Caller must pass the budget it is willing to spend; the default of 0
    means the function reports the cost and refuses to run.
    """
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
    X_samples = saltelli.sample(problem_def, n_base)

    # ---- cost gate: price the run before spending a cent ----
    n_runs = X_samples.shape[0]
    est_cost = n_runs * PILOT_COST_EUR
    print(f"n_base={n_base} expands to {n_runs} pilot runs "
          f"(Saltelli returns n_base*(2D+2), not n_base).")
    print(f"Estimated cost: {est_cost:,.0f} EUR. Authorised: {budget_eur:,.0f} EUR.")
    if est_cost > budget_eur:
        raise SystemExit(
            f"REFUSING TO RUN: {n_runs} pilot runs would cost {est_cost:,.0f} EUR, "
            f"but only {budget_eur:,.0f} EUR is authorised.\n"
            f"For reference the project benchmark spends 15,000 EUR IN TOTAL.\n"
            f"If you really mean it, pass budget_eur explicitly. To fit a budget "
            f"B, use\n  n_base <= B / ({PILOT_COST_EUR:.0f} * 12)  "
            f"-- e.g. 12,000 EUR allows n_base=0, i.e. this analysis does not fit.\n"
            f"A Sobol GSA on the real pilot scale is not affordable here; use the "
            f"surrogate\n(task5_sensitivity.py), or spend the budget on a 1-D "
            f"sweep of T/pH instead --\nsee task5_gsa_postmortem.md section 6.")

    # Store the real outcomes. NaN, not zeros: an unfilled row must poison the
    # analysis loudly rather than silently read as "Y = 0".
    Y_results = np.full(n_runs, np.nan)
    total_cost_gsa = 0.0

    # Evaluate each point with the API at pilot fidelity.
    for i, row in enumerate(X_samples):
        result = client.run("pilot", row[0], row[1], row[2], row[3], row[4])

        Y_results[i] = result['Y']
        total_cost_gsa += result['cost_eur']

        print(f"Run {i+1}/{len(X_samples)} finished. Current GSA cost: {total_cost_gsa} EUR")

    if np.isnan(Y_results).any():
        raise SystemExit(f"{int(np.isnan(Y_results).sum())} runs never returned "
                         f"a value; refusing to analyse a partial sample.")

    # Run Sobol analysis on the real outcomes.
    Si = sobol.analyze(problem_def, Y_results, print_to_console=False)
    
    print("\n--- GSA Ergebnisse ---")
    print(f"{'Variable':<10} | {'S1 (Haupteffekt)':<20} | {'ST (Gesamteffekt)':<20}")
    print("-" * 55)
    for i, name in enumerate(problem_def['names']):
        print(f"{name:<10} | {Si['S1'][i]:<20.4f} | {Si['ST'][i]:<20.4f}")

    check_sobol_validity(Si, problem_def['names'])
    return Si


def check_sobol_validity(Si, names):
    """Assert the identities a real Sobol estimate cannot violate.

    For independent inputs ST_i >= S1_i always, and sum(S1) <= sum(ST). A
    violation means the estimate is not measuring the process -- too few
    samples, a misaligned Y vector, or swapped columns -- and the indices must
    not be read as results. Printing this next to the table (rather than
    trusting the reader to check) is the guard that was missing when two of our
    GSAs disagreed.
    """
    s1 = np.asarray(Si['S1'], dtype=float)
    st = np.asarray(Si['ST'], dtype=float)
    bad = [n for n, a, b in zip(names, s1, st) if b < a]
    if bad or s1.sum() > st.sum():
        print("\n*** THESE INDICES ARE NOT VALID -- DO NOT REPORT THEM. ***")
        if bad:
            print(f"    ST < S1 for: {', '.join(bad)}. Sobol guarantees "
                  f"ST >= S1 for independent inputs.")
        if s1.sum() > st.sum():
            print(f"    sum(S1)={s1.sum():.3f} > sum(ST)={st.sum():.3f}, "
                  f"which is impossible.")
        print("    Likely causes: too few samples for the noise level; Y not "
              "aligned with the\n    sample matrix; or the S1/ST columns "
              "swapped where they are printed.")
    else:
        print(f"\n[validity ok] ST >= S1 for all inputs; "
              f"sum(S1)={s1.sum():.3f} <= sum(ST)={st.sum():.3f}.")
    return not bad

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


def get_client():
    """Authenticate lazily -- importing this module must not log in or spend."""
    c = BioreactorClient(BASE_URL)
    c.login(USER, PASSWORD)
    return c


def evaluate_final_model(csv_path, verify_with_api=False):
    """Fit the final model and return the best recommendation.

    verify_with_api=True runs the recommended recipe on the REAL pilot reactor
    via ``problem()`` -- one billed run at 2000 EUR. Off by default: this used
    to fire on every invocation of this script.
    """
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

    best_y_val = final_rec_y.item()
    best_y_val_denorm = scaler.denormalize(best_y_val, fidelity=1.0).item()

    mins = torch.tensor([r[0] for r in VARIABLE_RANGES.values()], **tkwargs)
    maxs = torch.tensor([r[1] for r in VARIABLE_RANGES.values()], **tkwargs)

    final_rec_x_denormal = mins + final_rec_x[0:5] * (maxs - mins)

    print(f"\n=== FINAL MODEL-ANALYSis ===")
    print(f"Best theorticall value (Pilot): {best_y_val_denorm:.3f}")
    print(f"Optimal recipe (denormalised):     {final_rec_x_denormal.squeeze().tolist()}")

    if verify_with_api:
        # One real pilot run at 2000 EUR -- opt-in only.
        print("Verifying the recommendation on the REAL pilot reactor "
              "(2000 EUR)...")
        final_rec_y_real, _ = problem(final_rec_x_full)
        print(f"Best real value (Pilot):        {final_rec_y_real.item():.3f}")

    return model, df, scaler





if __name__ == "__main__":
    print(__doc__)
    print("Run `python task5_sensitivity.py` for the maintained analysis.\n"
          "Nothing here runs by default -- every path below costs real money.")

    # The free, offline analysis. Superseded by task5_sensitivity.py, which
    # additionally gates each index on whether the design varied that input.
    #
    # DATEINAME = os.path.join("results",
    #                          "optimization_log_2026-07-11_09-43-48.csv")
    # model, daten_df, scaler = evaluate_final_model(DATEINAME)
    # run_global_sensitivity_analysis(model, scaler)

    # A real-API GSA. Left here as a deliberate, priced decision -- 384,000 EUR
    # at n_base=16 -- not as a default. See the cost gate in the function and
    # task5_gsa_postmortem.md section 6 for the affordable alternative.
    #
    # run_real_pilot_gsa(get_client(), n_base=16, budget_eur=0.0)
