import os
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
from mfbo_v1 import problem, save_step_to_csv
from basic_client import BioreactorClient
from scipy.stats import qmc
from datetime import datetime
from time import sleep

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

def generate_initial_data(n, micro, bench, pilot):
    # generate training data within the declared bounds
    train_x = torch.empty(n, 5, **tkwargs)


    # Uncomment the sampler you want to use (random / Latin Hypercube / Sobol)
    # # sample the 5 recipe variables from their actual ranges
    # for i, (low, high) in enumerate(VARIABLE_RANGES.values()):
    #     train_x[:, i] = torch.rand(n, **tkwargs) * (high - low) + low

    # # Latin Hypercube
    # sampler = qmc.LatinHypercube(d=5)
    # sample = sampler.random(n=n)  

    # Sobol
    sampler = qmc.Sobol(d=5)
    sample = sampler.random(n=n)

    train_x = torch.tensor(sample, **tkwargs)



    fidelities_list = [0]*micro + [0.5]*bench + [1]*pilot
    train_f_unshuffled = torch.tensor(fidelities_list, **tkwargs).unsqueeze(-1)
    
    shuffle_indices = torch.randperm(n)
    train_f = train_f_unshuffled[shuffle_indices]
    train_x_full = torch.cat((train_x, train_f), dim=1)
    
    train_obj, inital_cost = problem(train_x_full)  

    return train_x_full, train_obj, inital_cost

def evaluate_final_model(csv_path):
    print("Load data from csv file...")
    df = pd.DataFrame(pd.read_csv(csv_path))
    
    feature_cols = ["T", "pH_Wert", "F1", "F2", "F3", "Fidelity_Scaled"]
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
    print(f"Optimal recipe (denormalised):  {final_rec_x_denormal.squeeze().tolist()}")
    print(f"Total cost:                     {df.Cost_Cummulated.max()}")
    
    return model, df

if __name__ == "__main__":
    # run optimization
    target_fidelities = {5: 1}
    bounds = torch.tensor(
    [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], 
    **tkwargs
)
    # neter number of experiments for each scale
    micro = 5
    bench = 8
    pilot = 3
    n = micro+bench+pilot
    aktuelles_datum = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_datei_name = f"non_adaptive_log_{aktuelles_datum}.csv"
    print(f"This run will be saved in: results/{log_datei_name}")

    train_x, train_obj, cumulative_cost = generate_initial_data(n, micro, bench, pilot)
    running_cost = 0.0
    cost_mapping = {0: 10, 1: 500, 2: 2000}
    for idx in range(len(train_x)):
        row_x = train_x[idx]
        row_y = train_obj[idx]
        
        fid_idx = int(round(row_x[5].item() * 2.0))
        
        step_cost = cost_mapping[fid_idx]
        running_cost += step_cost
    
        save_step_to_csv(
            iteration=0, new_x=row_x, new_obj=row_y, cost=step_cost, 
            cumulative_cost=running_cost, pred_x=None, pred_y=None, 
            filename=log_datei_name
        )
    filepath = os.path.join("results", log_datei_name)
    evaluate_final_model(filepath)
    
