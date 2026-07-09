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

target_fidelities = {5: 1}

bounds = torch.tensor(
    [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], 
    **tkwargs)

SCALES = ["micro", "bench", "pilot"]

client = BioreactorClient(BASE_URL)
client.login(USER, PASSWORD)

fidelities = [0, 0.5, 1]
fidelities_costs = [0.85, 0.9, 1]
fidelities_real_costs = [10, 500, 2000]

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

    # # Latin Hypercube
    # sampler = qmc.LatinHypercube(d=5)
    # sample = sampler.random(n=n)  

    # Sobol
    sampler = qmc.Sobol(d=5, scramble=True)
    sample = sampler.random(n=n)

    train_x = torch.tensor(sample, **tkwargs)

    fidelities_list = [0.0]*59 + [0.5]*3 + [1.0]*2
    train_f_unshuffled = torch.tensor(fidelities_list, **tkwargs).unsqueeze(-1)
    
    shuffle_indices = torch.randperm(n)
    train_f = train_f_unshuffled[shuffle_indices]

    train_x_full = torch.cat((train_x, train_f), dim=1)
    
    train_obj, inital_cost = problem(train_x_full)  

    return train_x_full, train_obj, inital_cost

class CustomStaircaseCostModel(DeterministicModel):
    def __init__(self, fidelity_idx: int = -1, device=None, dtype=None) -> None:
        super().__init__()
        self.fidelity_idx = fidelity_idx
        self._num_outputs = 1
        
        costs = torch.tensor([0.01, 0.5, 2], device=device, dtype=dtype)
        self.register_buffer("costs", costs)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        fidelity = X[..., self.fidelity_idx]
        fidelity_indices = torch.clamp((fidelity * 2.0).round().long(), 0, 2)
        cost = self.costs[fidelity_indices]
        
        return cost.unsqueeze(-1)


class FidelityScaler:
    def __init__(self):
        self.stats = {}

    def _get_fidelity_values(self, train_x):
        fidelity = train_x[..., -1]
        return torch.clamp((fidelity * 2.0).round().long(), 0, 2)

    def _get_fidelity_key(self, fidelity_idx):
        return float([0.0, 0.5, 1.0][fidelity_idx])

    def get_stats(self, train_x, train_obj):
        train_obj_flat = train_obj.reshape(-1)
        fidelity_idx = self._get_fidelity_values(train_x)
        stats = {}

        for fid_idx in [0, 1, 2]:
            mask = fidelity_idx == fid_idx
            if mask.any():
                values = train_obj_flat[mask]
                min_val = values.min().item()
                max_val = values.max().item()
                stats[self._get_fidelity_key(fid_idx)] = (min_val, max_val)
            else:
                stats[self._get_fidelity_key(fid_idx)] = (0.0, 1.0)

        self.stats = stats
        return stats

    def transform(self, train_x, train_obj):
        stats = self.get_stats(train_x, train_obj)
        train_obj_flat = train_obj.reshape(-1)
        fidelity_idx = self._get_fidelity_values(train_x)
        transformed = torch.empty_like(train_obj_flat)

        for fid_idx in [0, 1, 2]:
            mask = fidelity_idx == fid_idx
            if not mask.any():
                continue
            min_val, max_val = stats[self._get_fidelity_key(fid_idx)]
            if abs(max_val - min_val) < 1e-12:
                transformed[mask] = torch.zeros(mask.sum(), **tkwargs)
            else:
                values = train_obj_flat[mask]
                transformed[mask] = (values - min_val) / (max_val - min_val)

        return transformed.reshape(train_obj.shape)

    def denormalize(self, values, fidelity=1.0):
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().reshape(-1)
        else:
            values = torch.tensor(values, **tkwargs).reshape(-1)

        if fidelity not in self.stats:
            return values

        min_val, max_val = self.stats[fidelity]
        if abs(max_val - min_val) < 1e-12:
            return torch.zeros_like(values)

        return values * (max_val - min_val) + min_val


def initialize_model(train_x, train_obj):
    model = SingleTaskMultiFidelityGP(
        train_x, train_obj, data_fidelities=[5]
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
    print('optimizing acqf')
    current_x, current_value = optimize_acqf(
        acq_function=curr_val_acqf,
        bounds=bounds[:, :-1],
        q=1,
        num_restarts=10 if not SMOKE_TEST else 2,
        raw_samples=256 if not SMOKE_TEST else 4,
        options={"batch_limit": 256, "maxiter": 200},
    )

    current_x_full = curr_val_acqf._construct_X_full(current_x)

    mfkg = qMultiFidelityKnowledgeGradient(
        model=model,
        num_fantasies=16 if not SMOKE_TEST else 2,
        current_value=current_value,
        cost_aware_utility=cost_aware_utility,
        project=project,
    )

    return mfkg, current_x_full, current_value

def optimize_mfkg_and_get_observation(mfkg_acqf):
    """Optimizes MFKG and returns a new candidate, observation, and cost."""

    # generate new candidates
    print('optimizing mfkg')
    candidates, _ = optimize_acqf_mixed(
        acq_function=mfkg_acqf,
        bounds=bounds,
        fixed_features_list=[{5: 0}, {5: 0.5}, {5: 1}],
        q=1,
        num_restarts=10 if not SMOKE_TEST else 2,
        raw_samples=256 if not SMOKE_TEST else 4,
        # batch_initial_conditions=X_init,
        options={"batch_limit": 256, "maxiter": 200},
    )

    # observe new values
    new_x = candidates.detach()
    return new_x

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

def save_step_to_csv(iteration, new_x, new_obj, cost, cumulative_cost, pred_x=None, pred_y=None, scaler=None, folder="results", filename="optimization_log.csv"):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    x_vals = new_x.squeeze().tolist()
    
    # Basis-Daten (wie bisher)
    row_dict = {
        "Iteration": iteration,
        "T": x_vals[0],
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
        if scaler is not None:
            y_val = scaler.denormalize(y_val, fidelity=1.0).item()
        
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

def baysian_optimization():
    # run optimization
    aktuelles_datum = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_datei_name = f"optimization_log_{aktuelles_datum}.csv"
    print(f"This run will be saved in: results/{log_datei_name}")

    train_x, train_obj, cumulative_cost = generate_initial_data(n=64)
    scaler = FidelityScaler()
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
            cumulative_cost=running_cost, pred_x=None, pred_y=None, scaler=scaler,
            filename=log_datei_name
        )

    N_ITER = 500 if not SMOKE_TEST else 3
    i=0
    micro_streak = 0
    schritte_ohne_pilot = 0
    try:
        while i < N_ITER and cumulative_cost<13000:
            train_obj_norm = scaler.transform(train_x, train_obj)
            mll, model = initialize_model(train_x, train_obj_norm)
            fit_gpytorch_mll(mll)
            mfkg_acqf, pred_opt_x, pred_opt_y = get_mfkg(model)
            
            # ensures a pilot run after 80 iterations
            if schritte_ohne_pilot >= 80:
                print(f"\n[Intervention] 80 runs without pilot. Enforce Pilot run at optimum")
                new_x = pred_opt_x.clone()
                new_x[0, 5] = 1.0  # Fidelity auf Pilot
                cumulative_cost += fidelities_real_costs[int(new_x[0,5]*2)]
                new_obj, cost = problem(new_x)
                
                # Zähler resetten (Ein Pilot-Versuch durchbricht auch jede Micro-Streak)
                schritte_ohne_pilot = 0 
                micro_streak = 0
                
            # ensures a bench run every 25 iterations
            elif micro_streak >= 25:
                print(f"\n[Intervention] 25 Micro-Attempts in row. Enforce Bench-calibration")
                new_x = pred_opt_x.clone()
                new_x[0, 5] = 0.5  
                cumulative_cost += fidelities_real_costs[int(new_x[0,5]*2)]
                new_obj, cost = problem(new_x)
                
                micro_streak = 0
                schritte_ohne_pilot += 1
                
                if schritte_ohne_pilot > 70:
                    schritte_ohne_pilot = 70
            
            # regular run without intervention
            else:
                new_x = optimize_mfkg_and_get_observation(mfkg_acqf)
                cumulative_cost += fidelities_real_costs[int(new_x[0,5]*2)]
                new_obj, cost = problem(new_x)
                print(f"candidate:{new_x}\n")
                print(f"observation:{new_obj}\n\n")
                # Prüfen, was die KI freiwillig gewählt hat
                new_fidelity = int(round(new_x[0, 5].item() * 2.0))
                
                if new_fidelity == 2:    # Pilot
                    schritte_ohne_pilot = 0
                    micro_streak = 0
                elif new_fidelity == 1:  # Bench
                    schritte_ohne_pilot += 1
                    micro_streak = 0
                else:                    # Micro
                    schritte_ohne_pilot += 1
                    micro_streak += 1

            train_x = torch.cat([train_x, new_x])
            train_obj = torch.cat([train_obj, new_obj])
            


            save_step_to_csv(
                iteration=i + 1, new_x=new_x, new_obj=new_obj, cost=cost, 
                cumulative_cost=cumulative_cost, pred_x=pred_opt_x, pred_y=pred_opt_y, scaler=scaler,
                filename=log_datei_name
            )
            i += 1
    except KeyboardInterrupt:
            print("\nOptimisation stopped by user!")
            print("Continue with model evaluation...")

    final_rec, final_y_val = get_recommendation(model)
    
    objective_value, cost = problem(final_rec)

    cumulative_cost += cost
    save_step_to_csv(
                iteration=i + 1, new_x=final_rec, new_obj=objective_value, cost=cost, 
                cumulative_cost=cumulative_cost, pred_x=None, pred_y=None, scaler=scaler,
                filename=log_datei_name
            )
    n_pilot = (train_x[:, 5] == 1.0).sum().item()+1

    torch.save(model.state_dict(), f"results/final_model_{aktuelles_datum}.pth")
    return objective_value, cumulative_cost, n_pilot

if __name__ == "__main__":
    N_TRIALS = 10
    trial_yields: list[float] = []
    trial_costs: list[float] = []
    trial_n_pilot: list[float] = []

    for trial in range(N_TRIALS):
        print(f"\n=== TRIAL {trial + 1} / {N_TRIALS} ===")
        Yield, cost, n_pilot = baysian_optimization()
        trial_yields.append(Yield.item())
        trial_costs.append(cost)
        trial_n_pilot.append(n_pilot)
        print(f"  -> Y = {Yield.item():.4f}")
        print(f"  -> Cost = {cost:.4f}")
        print(f"  -> n_pilot = {n_pilot:.4f}")

    print("\n=== SUMMARY OVER ALL TRIALS ===")
    print(f"  mean of trial means = {np.mean(trial_yields):.4f}")
    print(f"  variance of trial means = {np.var(trial_yields):.4f}")