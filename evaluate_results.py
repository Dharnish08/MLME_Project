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
from mfbo_v1 import problem
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
    print("Load data from csv file...")
    df = pd.DataFrame(pd.read_csv(csv_path))
    
    feature_cols = ["Temperatur", "pH_Wert", "F1", "F2", "F3", "Fidelity_Scaled"]
    train_x = torch.tensor(df[feature_cols].values, **tkwargs)
    train_obj = torch.tensor(df["Targetvalue_Y"].values, **tkwargs).unsqueeze(-1)
    
    print("Train final model with all data...")
    model = SingleTaskMultiFidelityGP(
        train_x, train_obj, outcome_transform=Standardize(m=1), data_fidelities=[5]
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
    best_y_val_real = final_rec_y_real.item()

    mins = torch.tensor([r[0] for r in VARIABLE_RANGES.values()], **tkwargs)
    maxs = torch.tensor([r[1] for r in VARIABLE_RANGES.values()], **tkwargs)

    final_rec_x_denormal = mins + final_rec_x[0:5] * (maxs - mins)

    print(f"\n=== FINAL MODEL-ANALYSis ===")
    print(f"Best theorticall value (Pilot): {best_y_val:.3f}")
    print(f"Best real value (Pilot):        {best_y_val_real:.3f}")
    print(f"Optimal recipe (denormalised):     {final_rec_x_denormal.squeeze().tolist()}")
    
    return model, df

def plot_fast_convergence(df):

    print("\nStart evaluating previous results...")
    
    df['Pilot_Y_Only'] = df['Targetvalue_Y'].where(df['Fidelity_Index'] == 2)
    
    df['Running_Max_Y'] = df['Pilot_Y_Only'].cummax().ffill()
    
    df['Running_Max_Y'] = df['Running_Max_Y'].bfill()
    
    df_opt = df[df['Iteration'] > 0].copy()
    
    cumulative_costs = df_opt["Cost_Cummulated"].values
    predicted_max_y = df_opt["Pred_Max_Y"].values
    real_max_y = df_opt["Running_Max_Y"].values
    
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
    
    plt.plot(cumulative_costs, predicted_max_y, label='Model prediction (Theoretical Maximum)', 
             color='blue', linewidth=2, linestyle='-')
             
    plt.plot(cumulative_costs, true_recommendation_y, label='Real value at predicted maximum', 
             color='red', linewidth=2, linestyle='-')
    
    plt.step(cumulative_costs, real_max_y, label='actual previous measured maximum', 
             color='black', linewidth=2, linestyle='--', where='post')
    
    plt.xlabel('cummulated cost', fontsize=12)
    plt.ylabel('Target value $Y$', fontsize=12)
    plt.title('Model-convergence', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Trage hier den korrekten Namen deiner aktuellsten Log-Datei ein!
    DATEINAME = "results\optimization_log_2026-06-24_00-32-57.csv"
    
    model, daten_df = evaluate_final_model(DATEINAME)
    plot_fast_convergence(daten_df)