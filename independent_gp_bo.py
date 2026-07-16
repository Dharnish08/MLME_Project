import os
from datetime import datetime

import pandas as pd
import torch
from scipy.stats import qmc

from basic_client import BASE_URL, PASSWORD, USER, BioreactorClient
from data_utils import VARIABLE_RANGES, tkwargs
from independent_gp import IndependentPerScaleGP

SCALES = ["micro", "bench", "pilot"]
COST_OF_SCALE = {"micro": 10.0, "bench": 500.0, "pilot": 2000.0}  # mfbo_v1.py's cost_mapping
INITIAL_BATCH = {"micro": 56, "bench": 6, "pilot": 2}  # mfbo_v1.py's generate_initial_data split

EXPLORATION_BUDGET_EUR = 13000.0  # covers the initial batch too, same as mfbo_v1.py
MAX_ITERS = 300
MICRO_STREAK_LIMIT = 10
PILOT_GAP_LIMIT = 5

_NORMAL = torch.distributions.Normal(0.0, 1.0)


def expected_improvement(mean, var, best_so_far):
    std = var.clamp_min(1e-10).sqrt()
    z = (mean - best_so_far) / std
    return (std * (z * _NORMAL.cdf(z) + torch.exp(_NORMAL.log_prob(z)))).clamp_min(0.0)


def pick_next_query(model, best_pilot_y, n_candidates=4096):
    candidates = torch.tensor(
        qmc.Sobol(d=len(VARIABLE_RANGES), scramble=True).random(n=n_candidates), **tkwargs
    )
    best = None
    for scale in SCALES:
        mean, var = model.posterior(candidates, level=scale)
        score = expected_improvement(mean, var, best_pilot_y) / COST_OF_SCALE[scale]
        idx = score.argmax()
        if best is None or score[idx] > best[0]:
            best = (score[idx], candidates[idx : idx + 1], scale)
    _, x, scale = best
    return x, scale


def best_pilot_guess(model, n_candidates=16384):
    candidates = torch.tensor(
        qmc.Sobol(d=len(VARIABLE_RANGES), scramble=True, seed=0).random(n=n_candidates), **tkwargs
    )
    mean, _ = model.posterior(candidates, level="pilot")
    return candidates[mean.argmax() : mean.argmax() + 1]


def denormalize(x):
    values = x.squeeze(0).tolist()
    return {k: lo + v * (hi - lo) for v, (k, (lo, hi)) in zip(values, VARIABLE_RANGES.items())}


def record(data, scale, x, y_value):
    x_prev, y_prev = data[scale]
    data[scale] = (torch.cat([x_prev, x]), torch.cat([y_prev, torch.tensor([[y_value]], **tkwargs)]))


def fit(data):
    try:
        return IndependentPerScaleGP(data)
    except Exception as exc:  # noqa: BLE001
        print(f"  (fit failed once: {exc!r} - retrying)")
        return IndependentPerScaleGP(data)


def run_initial_batch(client):
    n = sum(INITIAL_BATCH.values())
    x_all = torch.tensor(qmc.Sobol(d=len(VARIABLE_RANGES), scramble=True).random(n=n), **tkwargs)
    scales = [s for s, count in INITIAL_BATCH.items() for _ in range(count)]
    scales = [scales[i] for i in torch.randperm(n)]

    data = {s: (torch.empty(0, len(VARIABLE_RANGES), **tkwargs), torch.empty(0, 1, **tkwargs)) for s in SCALES}
    spent = 0.0
    for i, scale in enumerate(scales):
        x = x_all[i : i + 1]
        result = client.run(scale, **denormalize(x))
        record(data, scale, x, result["Y"])
        spent += result["cost_eur"]
        print(f"init {i + 1}/{n} scale={scale} Y={result['Y']:.3f} spent={spent:.0f}")

    return data, spent


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    total_cap = EXPLORATION_BUDGET_EUR + COST_OF_SCALE["pilot"]
    if input("Type 'yes' to spend real budget and run this campaign: ").strip().lower() != "yes":
        raise SystemExit("Cancelled - nothing spent.")

    client = BioreactorClient(BASE_URL)
    client.login(USER, PASSWORD)

    print("\n--- Initial batch ---")
    data, spent = run_initial_batch(client)
    best_pilot_y = data["pilot"][1].max().item()
    print(f"Initial batch done: {spent:.0f} EUR spent, best pilot Y so far = {best_pilot_y:.3f}")

    micro_streak = 0
    iters_since_pilot = 0
    n_pilot_evals = 0
    log = []

    # --- Phase 1: adaptive exploration, hard-capped at 13,000 EUR total ---
    print("\n--- Adaptive exploration ---")
    it = 0
    while it < MAX_ITERS:
        it += 1
        model = fit(data)

        if iters_since_pilot >= PILOT_GAP_LIMIT:
            print(f"it={it}: forcing a pilot confirmation")
            x_next, scale_next = best_pilot_guess(model), "pilot"
        elif micro_streak >= MICRO_STREAK_LIMIT:
            print(f"it={it}: forcing a bench confirmation")
            x_next, scale_next = best_pilot_guess(model), "bench"
        else:
            x_next, scale_next = pick_next_query(model, best_pilot_y)

        cost = COST_OF_SCALE[scale_next]
        if spent + cost > EXPLORATION_BUDGET_EUR:
            print(f"it={it}: next query ({scale_next}) would push exploration spend "
                  f"past {EXPLORATION_BUDGET_EUR:.0f} EUR (currently {spent:.0f}) - stopping")
            it -= 1
            break

        recipe = denormalize(x_next)
        result = client.run(scale_next, **recipe)
        record(data, scale_next, x_next, result["Y"])
        spent += result["cost_eur"]

        if scale_next == "pilot":
            n_pilot_evals += 1
            iters_since_pilot = 0
            micro_streak = 0
            best_pilot_y = max(best_pilot_y, result["Y"])
        elif scale_next == "bench":
            iters_since_pilot += 1
            micro_streak = 0
        else:
            iters_since_pilot += 1
            micro_streak += 1

        print(f"it={it} scale={scale_next} Y={result['Y']:.3f} "
              f"spent={spent:.0f}/{EXPLORATION_BUDGET_EUR:.0f} best_pilot={best_pilot_y:.3f}")
        log.append({
            "iteration": it, "phase": "exploration", "scale": scale_next,
            "Y": result["Y"], "cost_eur": result["cost_eur"],
            "cumulative_cost": spent, "best_pilot_y_so_far": best_pilot_y, **recipe,
        })

    print(f"\nExploration done: {it} adaptive iterations on top of the initial batch, "
          f"{spent:.0f} EUR spent in total, best pilot Y so far = {best_pilot_y:.3f}")

    # --- Phase 2: spend the reserved final confirmation on our best guess ---
    model = fit(data)
    final_x = best_pilot_guess(model)
    predicted_y = model.posterior(final_x, level="pilot")[0].item()
    final_recipe = denormalize(final_x)

    print(f"Spending the reserved final pilot confirmation on our best recipe "
          f"(predicted Y = {predicted_y:.3f})")
    result = client.run("pilot", **final_recipe)
    spent += result["cost_eur"]
    n_pilot_evals += 1
    best_pilot_y = max(best_pilot_y, result["Y"])
    record(data, "pilot", final_x, result["Y"])

    log.append({
        "iteration": it + 1, "phase": "final_confirmation", "scale": "pilot",
        "Y": result["Y"], "predicted_y": predicted_y, "cost_eur": result["cost_eur"],
        "cumulative_cost": spent, "best_pilot_y_so_far": best_pilot_y, **final_recipe,
    })

    print(f"\nFinal recipe: predicted Y = {predicted_y:.3f}, real Y = {result['Y']:.3f}")
    print(f"Best pilot Y this campaign: {best_pilot_y:.3f}")
    print(f"Total spend: {spent:.0f} EUR across {n_pilot_evals} pilot evaluations")

    log_path = f"results/independent_gp_bo_{timestamp}.csv"
    pd.DataFrame(log).to_csv(log_path, index=False)
    print(f"Saved {log_path}")