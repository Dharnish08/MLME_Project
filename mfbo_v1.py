import os
import torch
from botorch.test_functions.multi_fidelity import AugmentedHartmann
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_mll
from botorch.models.deterministic import DeterministicModel
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition import PosteriorMean
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition import qLogExpectedImprovement
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.optim.optimize import optimize_acqf_mixed
import pandas as pd
import numpy as np
from basic_client import BioreactorClient
import time
from scipy.stats import qmc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from datetime import datetime


SMOKE_TEST = False

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

USER = "group14"
PASSWORD = "ip00y-p92nc-9smqo"
BASE_URL = "https://mlme26biosim.org"

# Parameter

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

fidelities = [0, 0.5, 1]
fidelities_costs = [0.01, 0.5, 2]

def problem(args):
    args_2d = args.view(-1, args.shape[-1])
    
    # Tensoren für die Skalierung bereitstellen
    mins = torch.tensor([r[0] for r in VARIABLE_RANGES.values()], **tkwargs)
    maxs = torch.tensor([r[1] for r in VARIABLE_RANGES.values()], **tkwargs)

    y_values = []
    total_cost = 0.0
    
    for row in args_2d:
        fidelity_index = torch.clamp((row[5] * 2.0).round().long(), 0, 2).item()
        scale = SCALES[fidelity_index]

        api_inputs = mins + row[0:5] * (maxs - mins)

        result = client.run(scale, *api_inputs[0:5].tolist())
        print(result)
        
        y_values.append(result['Y'])
        total_cost += result['cost_eur']
        

    result_Y = torch.tensor(y_values, **tkwargs)
    
    if args.ndim > 1:
        result_Y = result_Y.unsqueeze(-1)
    else:
        result_Y = result_Y[0]
        
    return result_Y, total_cost

def generate_initial_data(n):
    # generate training data within the declared bounds
    train_x = torch.empty(n, 5, **tkwargs)

    # # sample the 5 recipe variables from their actual ranges
    # for i, (low, high) in enumerate(VARIABLE_RANGES.values()):
    #     train_x[:, i] = torch.rand(n, **tkwargs) * (high - low) + low

    # Latin Hypercube
    sampler = qmc.LatinHypercube(d=5)
    sample = sampler.random(n=n)  

    train_x = torch.tensor(sample, **tkwargs)


    # train_f = torch.tensor(fidelities, **tkwargs)[torch.randint(len(fidelities), (n, 1))]
    train_f = torch.tensor([0]*8 + [0.5]*3 + [1]*1, **tkwargs).unsqueeze(-1)
    train_x_full = torch.cat((train_x, train_f), dim=1)
    
    train_obj, inital_cost = problem(train_x_full)  

    return train_x_full, train_obj, inital_cost

class CustomStaircaseCostModel(DeterministicModel):
    def __init__(self, fidelity_idx: int = -1, device=None, dtype=None) -> None:
        super().__init__()
        self.fidelity_idx = fidelity_idx
        self._num_outputs = 1
        
        costs = torch.tensor([0.01, 0.075, 0.5], device=device, dtype=dtype)
        self.register_buffer("costs", costs)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        fidelity = X[..., self.fidelity_idx]
        fidelity_indices = torch.clamp((fidelity * 2.0).round().long(), 0, 2)
        cost = self.costs[fidelity_indices]
        
        return cost.unsqueeze(-1)

def initialize_model(train_x, train_obj):
    model = SingleTaskMultiFidelityGP(
        train_x, train_obj, outcome_transform=Standardize(m=1), data_fidelities=[5]
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model

def project(X):
    return project_to_target_fidelity(X=X, target_fidelities=target_fidelities, d=6)

def get_mfkg(model):
    cost_model = CustomStaircaseCostModel(fidelity_idx=5, **tkwargs)
    cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

    curr_val_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model, maximize=True),
        d=6, columns=[5], values=[1],
    )

    current_x, current_value = optimize_acqf(
        acq_function=curr_val_acqf,
        bounds=bounds[:, :-1],
        q=1,
        num_restarts=10 if not SMOKE_TEST else 2,
        raw_samples=256 if not SMOKE_TEST else 4,
        options={"batch_limit": 128, "maxiter": 200},
    )

    current_x_full = curr_val_acqf._construct_X_full(current_x)

    mfkg = qMultiFidelityKnowledgeGradient(
        model=model,
        num_fantasies=32 if not SMOKE_TEST else 2,
        current_value=current_value,
        cost_aware_utility=cost_aware_utility,
        project=project,
    )

    return mfkg, current_x_full, current_value

def optimize_mfkg_and_get_observation(mfkg_acqf):
    """Optimizes MFKG and returns a new candidate, observation, and cost."""

    # generate new candidates
    candidates, _ = optimize_acqf_mixed(
        acq_function=mfkg_acqf,
        bounds=bounds,
        fixed_features_list=[{5: 0}, {5: 0.5}, {5: 1}],
        q=1,
        num_restarts=10 if not SMOKE_TEST else 2,
        raw_samples=256 if not SMOKE_TEST else 4,
        # batch_initial_conditions=X_init,
        options={"batch_limit": 128, "maxiter": 200},
    )

    # observe new values
    new_x = candidates.detach()
    new_obj, cost = problem(new_x)
    print(f"candidate:{new_x}\n")
    print(f"observation:{new_obj}\n\n")
    return new_x, new_obj, cost

def get_recommendation(model):
    rec_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model, maximize=True),
        d=6,
        columns=[5],
        values=[1],
    )

    final_rec, expected_value = optimize_acqf(
        acq_function=rec_acqf,
        bounds=bounds[:, :-1],
        q=1,
        num_restarts=10,
        raw_samples=512,
        options={"batch_limit": 128, "maxiter": 200},
    )

    final_rec = rec_acqf._construct_X_full(final_rec)

    print(f"recommended point:{final_rec}\nexpected value:{expected_value}")
    return final_rec, expected_value

def save_step_to_csv(iteration, new_x, new_obj, cost, cumulative_cost, pred_x=None, pred_y=None, folder="results", filename="optimization_log.csv"):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    x_vals = new_x.squeeze().tolist()
    
    # Basis-Daten (wie bisher)
    row_dict = {
        "Iteration": iteration,
        "Temperatur": x_vals[0],
        "pH_Wert": x_vals[1],
        "F1": x_vals[2],
        "F2": x_vals[3],
        "F3": x_vals[4],
        "Fidelity_Scaled": x_vals[5],
        "Fidelity_Index": int(round(x_vals[5] * 2.0)),
        "Targetvalue_Y": new_obj.item(),
        "Step_Cost": cost,
        "Cost_Cummulated": cumulative_cost
    }

    if pred_x is not None and pred_y is not None:
        p_vals = pred_x.squeeze().tolist()
        
        y_val = pred_y.item() if torch.is_tensor(pred_y) else pred_y
        
        row_dict.update({
            "Pred_Opt_T": p_vals[0],
            "Pred_Opt_pH": p_vals[1],
            "Pred_Opt_F1": p_vals[2],
            "Pred_Opt_F2": p_vals[3],
            "Pred_Opt_F3": p_vals[4],
            "Pred_Max_Y": y_val
        })
    else:
        row_dict.update({
            "Pred_Opt_T": "", "Pred_Opt_pH": "", "Pred_Opt_F1": "", 
            "Pred_Opt_F2": "", "Pred_Opt_F3": "", "Pred_Max_Y": ""
        })

    df = pd.DataFrame([row_dict])
    file_exists = os.path.isfile(filepath)
    df.to_csv(filepath, mode='a', index=False, header=not file_exists)

if __name__ == "__main__":
    # run optimization
    target_fidelities = {5: 1}
    bounds = torch.tensor(
    [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], 
    **tkwargs
)
    aktuelles_datum = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_datei_name = f"optimization_log_{aktuelles_datum}.csv"
    print(f"This run will be saved in: results/{log_datei_name}")

    train_x, train_obj, cumulative_cost = generate_initial_data(n=12)
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

    N_ITER = 150 if not SMOKE_TEST else 3
    i=0
    try:
        while i < N_ITER and cumulative_cost<15000:
            mll, model = initialize_model(train_x, train_obj)
            fit_gpytorch_mll(mll)
            mfkg_acqf, pred_opt_x, pred_opt_y = get_mfkg(model)
            new_x, new_obj, cost = optimize_mfkg_and_get_observation(mfkg_acqf)
            train_x = torch.cat([train_x, new_x])
            train_obj = torch.cat([train_obj, new_obj])
            
            cumulative_cost += cost

            save_step_to_csv(
                iteration=i + 1, new_x=new_x, new_obj=new_obj, cost=cost, 
                cumulative_cost=cumulative_cost, pred_x=pred_opt_x, pred_y=pred_opt_y, 
                filename=log_datei_name
            )
            i += 1
    except KeyboardInterrupt:
            print("\nOptimisation stopped by user!")
            print("Continue with model evaluation...")

    final_rec, final_y_val = get_recommendation(model)
    
    objective_value, cost = problem(final_rec)
