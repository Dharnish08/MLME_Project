"""Additional task 3: does the cheap scale information actually help?

We compare, at *equal total cost*, the project's multi-scale strategy against a
single-scale (pilot-only) Bayesian-optimization loop.  Everything runs offline
against a frozen ground-truth oracle, so no API budget is spent.

Setup
-----
* Oracle : a ``SingleTaskMultiFidelityGP`` fitted once on *all* logged
  experiments and then frozen.  Its posterior mean is the "true" process
  response Y(recipe, scale); per-scale Gaussian noise is added on every query.
  Every arm is scored against this one identical ground truth.

Three arms, all given the same budget (~13 230 EUR = the campaign's cost):

* Multi-scale BO (live) -- THE FAIR COMPARISON.  ``mfbo_v1.py``'s actual
  strategy, from scratch: its own 64-run Sobol initial design (56 micro /
  6 bench / 2 pilot), then a cost-aware ``qMultiFidelityKnowledgeGradient``
  loop that picks its own recipes AND scales, with the same two interventions
  (force bench after 20 micros; force pilot after 80 steps without one).
* Pilot-only BO (live) : an adaptive loop that may ONLY run pilot experiments
  (2000 EUR each), so the budget buys floor(budget/2000) = 6 runs.  Single-
  fidelity GP + LogEI.
* Campaign replay (biased reference) : re-runs the finished campaign's exact
  (recipe, scale) decisions against the oracle.  This does NOT search -- it
  replays recipes the real BO loop already chose, and the oracle is fitted to
  that same campaign, so the oracle peaks where the replay is already headed.
  It reproduces the ceiling almost exactly and is reported only as an UPPER
  BOUND on what cheap-scale data could buy, never as the result.

* Score : the oracle's true (noise-free) pilot yield at each arm's recommended
  recipe, plus the best pilot yield actually observed.  Repeated over seeds;
  the two live arms are compared paired per seed.

If the cheap micro/bench data genuinely sharpen the pilot surrogate, the live
multi-scale arm should recommend a better recipe than a pilot-only loop that
spent the same money on a handful of expensive pilot runs.

The live arm is slow (MFKG over a growing GP), so seeds run in parallel
processes; expect tens of minutes.

    python task3_pilot_vs_multiscale.py [--seeds N] [--budget EUR] [--jobs N]
"""
import argparse
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
from botorch import fit_gpytorch_mll
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.acquisition import PosteriorMean
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.models import SingleTaskGP
from botorch.models.deterministic import DeterministicModel
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim.optimize import optimize_acqf, optimize_acqf_mixed
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from scipy.stats import qmc

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "results", "tables_initial_experiments",
                   "cleaned_full_optimization_log_reference.csv")
FIG_DIR = os.path.join(HERE, "results", "figures_additional_tasks")
TAB_DIR = os.path.join(HERE, "results", "tables_additional_tasks")

TKW = {"dtype": torch.double, "device": torch.device("cpu")}
VARS = ["T", "pH", "F1", "F2", "F3"]
NORM_COLS = ["T_norm", "pH_norm", "F1_norm", "F2_norm", "F3_norm"]
COST = {"micro": 10, "bench": 500, "pilot": 2000}
FID_OF_SCALE = {"micro": 0.0, "bench": 0.5, "pilot": 1.0}
BOUNDS5 = torch.tensor([[0.0] * 5, [1.0] * 5], **TKW)
BOUNDS6 = torch.tensor([[0.0] * 6, [1.0] * 6], **TKW)


def SCALE_OF_FID(f: float) -> str:
    """Scale name for a fidelity value; rounds as mfbo_v1's cost model does."""
    return ["micro", "bench", "pilot"][int(np.clip(round(float(f) * 2), 0, 2))]


# --------------------------------------------------------------------------- #
#  Oracle (frozen ground truth)                                               #
# --------------------------------------------------------------------------- #
def build_oracle(df: pd.DataFrame):
    X = torch.tensor(df[NORM_COLS].values, **TKW)
    f = torch.tensor(df["Fidelity_Scaled"].values, **TKW).unsqueeze(-1)
    Y = torch.tensor(df["Y"].values, **TKW).unsqueeze(-1)
    X6 = torch.cat([X, f], dim=1)
    model = SingleTaskMultiFidelityGP(
        X6, Y, outcome_transform=Standardize(m=1), data_fidelities=[5])
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    model.eval()

    # per-scale observation noise = std of (observed - oracle mean) residuals,
    # floored so a near-interpolating GP does not report ~0 noise.
    with torch.no_grad():
        mu = model.posterior(X6).mean.squeeze(-1)
    resid = (Y.squeeze(-1) - mu).numpy()
    noise = {}
    for sc in ["micro", "bench", "pilot"]:
        m = df["scale"].to_numpy() == sc
        floor = 0.02 * (df.loc[m, "Y"].std() if m.sum() > 1 else 1.0)
        noise[sc] = float(max(resid[m].std() if m.sum() > 1 else 0.0, floor))
    return model, noise


def oracle_mean(model, U: np.ndarray, fidelity: float) -> np.ndarray:
    X6 = np.hstack([U, np.full((U.shape[0], 1), fidelity)])
    with torch.no_grad():
        return model.posterior(torch.tensor(X6, **TKW)).mean.squeeze(-1).numpy()


def oracle_observe(model, u: np.ndarray, scale: str, rng) -> float:
    mu = oracle_mean(model, u.reshape(1, -1), FID_OF_SCALE[scale])[0]
    return float(mu + rng.normal(0.0, model_noise[scale]))


# --------------------------------------------------------------------------- #
#  Recommendation helpers                                                      #
# --------------------------------------------------------------------------- #
def recommend_multifidelity(model) -> np.ndarray:
    """argmax of the posterior mean at pilot fidelity (as in mfbo_v1)."""
    acqf = FixedFeatureAcquisitionFunction(
        PosteriorMean(model, maximize=True), d=6, columns=[5], values=[1.0])
    x, _ = optimize_acqf(acqf, bounds=BOUNDS5, q=1,
                         num_restarts=5, raw_samples=128)
    return x.detach().numpy().ravel()


def recommend_singletask(model) -> np.ndarray:
    x, _ = optimize_acqf(PosteriorMean(model, maximize=True), bounds=BOUNDS5,
                         q=1, num_restarts=5, raw_samples=128)
    return x.detach().numpy().ravel()


# --------------------------------------------------------------------------- #
#  Live multi-fidelity arm: mfbo_v1's actual strategy, run against the oracle  #
# --------------------------------------------------------------------------- #
#  Mirrors mfbo_v1.py: a SingleTaskMultiFidelityGP over per-fidelity min-max
#  scaled Y, a cost-aware qMultiFidelityKnowledgeGradient, and the same two
#  interventions.  Unlike run_multiscale() it picks its own recipes AND scales,
#  so it can be compared with the pilot-only arm on equal terms.
MICRO_STREAK_LIMIT = 20
NO_PILOT_LIMIT = 80
INIT_FIDELITIES = [0.0] * 56 + [0.5] * 6 + [1.0] * 2   # mfbo_v1's 64-run design


class StaircaseCostModel(DeterministicModel):
    """mfbo_v1's cost model: cost per scale, in thousands of EUR."""

    def __init__(self, fidelity_idx: int = 5):
        super().__init__()
        self.fidelity_idx = fidelity_idx
        self._num_outputs = 1
        self.register_buffer("costs", torch.tensor([0.01, 0.5, 2.0], **TKW))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        fid = X[..., self.fidelity_idx]
        idx = torch.clamp((fid * 2.0).round().long(), 0, 2)
        return self.costs[idx].unsqueeze(-1)


def scale_per_fidelity(X6: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Per-fidelity min-max scaling of Y, as mfbo_v1's FidelityScaler does."""
    fid_idx = torch.clamp((X6[..., 5] * 2.0).round().long(), 0, 2)
    out = torch.empty_like(Y.reshape(-1))
    flat = Y.reshape(-1)
    for k in [0, 1, 2]:
        m = fid_idx == k
        if not m.any():
            continue
        v = flat[m]
        lo, hi = v.min(), v.max()
        out[m] = torch.zeros(int(m.sum()), **TKW) if abs(hi - lo) < 1e-12 \
            else (v - lo) / (hi - lo)
    return out.reshape(Y.shape)


def _fit_mf(X6: torch.Tensor, Y: torch.Tensor) -> SingleTaskMultiFidelityGP:
    model = SingleTaskMultiFidelityGP(X6, scale_per_fidelity(X6, Y),
                                      data_fidelities=[5])
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    return model


def _mfkg(model, fantasies: int, restarts: int, raw: int):
    """Cost-aware MFKG plus the current predicted pilot optimum."""
    curr = FixedFeatureAcquisitionFunction(
        PosteriorMean(model, maximize=True), d=6, columns=[5], values=[1.0])
    cur_x, cur_val = optimize_acqf(curr, bounds=BOUNDS5, q=1,
                                   num_restarts=restarts, raw_samples=raw,
                                   options={"batch_limit": 64, "maxiter": 120})
    acqf = qMultiFidelityKnowledgeGradient(
        model=model, num_fantasies=fantasies, current_value=cur_val,
        cost_aware_utility=InverseCostWeightedUtility(
            cost_model=StaircaseCostModel(fidelity_idx=5)),
        project=lambda X: project_to_target_fidelity(
            X=X, target_fidelities={5: 1.0}, d=6),
    )
    return acqf, curr._construct_X_full(cur_x)


def run_multiscale_live(oracle, budget: float, rng, max_iter: int = 200,
                        fantasies: int = 8, restarts: int = 3, raw: int = 64,
                        verbose: bool = False):
    """mfbo_v1's strategy, choosing its own recipes and scales from scratch."""
    sob = qmc.Sobol(d=5, scramble=True, seed=int(rng.integers(1 << 31)))
    U = sob.random(len(INIT_FIDELITIES))
    fids = np.array(INIT_FIDELITIES)[rng.permutation(len(INIT_FIDELITIES))]
    scales = [SCALE_OF_FID(f) for f in fids]
    y = np.array([oracle_observe(oracle, U[i], scales[i], rng)
                  for i in range(len(U))])

    cost = float(sum(COST[s] for s in scales))
    best = max([y[i] for i in range(len(y)) if scales[i] == "pilot"],
               default=-np.inf)
    curve = [(cost, best)]
    n_forced = {"pilot": 0, "bench": 0}
    micro_streak = steps_without_pilot = 0

    it = 0
    while cost < budget and it < max_iter:
        X6 = torch.tensor(np.hstack([U, fids.reshape(-1, 1)]), **TKW)
        Yt = torch.tensor(y.reshape(-1, 1), **TKW)
        model = _fit_mf(X6, Yt)
        acqf, pred_opt = _mfkg(model, fantasies, restarts, raw)

        # Same intervention ladder as mfbo_v1, in the same order.
        if steps_without_pilot >= NO_PILOT_LIMIT:
            u_new, f_new = pred_opt.detach().numpy().ravel()[:5], 1.0
            steps_without_pilot = micro_streak = 0
            n_forced["pilot"] += 1
        elif micro_streak >= MICRO_STREAK_LIMIT:
            u_new, f_new = pred_opt.detach().numpy().ravel()[:5], 0.5
            micro_streak = 0
            steps_without_pilot = min(steps_without_pilot + 1, 70)
            n_forced["bench"] += 1
        else:
            cand, _ = optimize_acqf_mixed(
                acq_function=acqf, bounds=BOUNDS6,
                fixed_features_list=[{5: 0.0}, {5: 0.5}, {5: 1.0}], q=1,
                num_restarts=restarts, raw_samples=raw,
                options={"batch_limit": 64, "maxiter": 120})
            v = cand.detach().numpy().ravel()
            u_new, f_new = v[:5], float(v[5])
            k = int(round(f_new * 2))
            if k == 2:
                steps_without_pilot = micro_streak = 0
            elif k == 1:
                steps_without_pilot += 1
                micro_streak = 0
            else:
                steps_without_pilot += 1
                micro_streak += 1

        # Hard budget: never take a run we cannot afford. `while cost < budget`
        # alone lets a final 500/2000 EUR run overshoot, which would hand this
        # arm more money than the pilot-only arm and break the equal-cost
        # premise. (mfbo_v1 itself overshoots this way; here it must not.)
        sc = SCALE_OF_FID(f_new)
        if cost + COST[sc] > budget:
            break

        y_new = oracle_observe(oracle, u_new, sc, rng)
        U = np.vstack([U, u_new])
        fids = np.append(fids, f_new)
        y = np.append(y, y_new)
        cost += COST[sc]
        if sc == "pilot":
            best = max(best, y_new)
        curve.append((cost, best))
        it += 1
        if verbose and it % 10 == 0:
            print(f"    [live] iter {it:3d}  cost {cost:7.0f}/{budget:.0f}  "
                  f"best pilot {best:.2f}", flush=True)

    X6 = torch.tensor(np.hstack([U, fids.reshape(-1, 1)]), **TKW)
    model = _fit_mf(X6, torch.tensor(y.reshape(-1, 1), **TKW))
    model.eval()
    rec = recommend_multifidelity(model)
    return {"cost": cost, "n_iter": it, "best_pilot_obs": float(best),
            "curve": curve, "rec": rec,
            "rec_true": float(oracle_mean(oracle, rec.reshape(1, -1), 1.0)[0]),
            "n_forced": n_forced,
            "n_by_scale": {s: int(sum(1 for f in fids
                                      if SCALE_OF_FID(f) == s))
                           for s in ["micro", "bench", "pilot"]}}


# --------------------------------------------------------------------------- #
#  The replay arm (kept as a biased reference)                                 #
# --------------------------------------------------------------------------- #
def run_multiscale(oracle, campaign: pd.DataFrame, rng):
    """Replay the real campaign against the oracle; fit the final surrogate."""
    U = campaign[NORM_COLS].to_numpy()
    scales = campaign["scale"].to_numpy()
    fids = campaign["Fidelity_Scaled"].to_numpy()

    cost, best_pilot, curve = 0.0, -np.inf, []
    Yobs = np.empty(len(U))
    for i in range(len(U)):
        y = oracle_observe(oracle, U[i], scales[i], rng)
        Yobs[i] = y
        cost += COST[scales[i]]
        if scales[i] == "pilot":
            best_pilot = max(best_pilot, y)
        curve.append((cost, best_pilot))

    X6 = torch.tensor(np.hstack([U, fids.reshape(-1, 1)]), **TKW)
    Y = torch.tensor(Yobs.reshape(-1, 1), **TKW)
    model = SingleTaskMultiFidelityGP(
        X6, Y, outcome_transform=Standardize(m=1), data_fidelities=[5])
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    model.eval()
    rec = recommend_multifidelity(model)
    rec_true = float(oracle_mean(oracle, rec.reshape(1, -1), 1.0)[0])
    return {"cost": cost, "best_pilot_obs": best_pilot, "curve": curve,
            "rec": rec, "rec_true": rec_true}


def run_pilot_only(oracle, budget: float, rng, n_init: int = 2):
    """Adaptive pilot-only BO: n_init Sobol pilots + greedy LogEI pilots."""
    sob = qmc.Sobol(d=5, scramble=True, seed=int(rng.integers(1 << 31)))
    U = sob.random(n_init)
    y = np.array([oracle_observe(oracle, U[i], "pilot", rng)
                  for i in range(n_init)])

    # Credit each initial pilot only once it has actually been paid for: after
    # k runs the curve may show the best of the first k, not the best of all
    # n_init.
    curve = [(float((k + 1) * COST["pilot"]), float(y[:k + 1].max()))
             for k in range(n_init)]
    cost = n_init * COST["pilot"]
    best = y.max()

    while cost + COST["pilot"] <= budget:
        Xt = torch.tensor(U, **TKW)
        Yt = torch.tensor(y.reshape(-1, 1), **TKW)
        model = SingleTaskGP(Xt, Yt, outcome_transform=Standardize(m=1))
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
        acqf = LogExpectedImprovement(model, best_f=float(y.max()),
                                      maximize=True)
        cand, _ = optimize_acqf(acqf, bounds=BOUNDS5, q=1,
                                num_restarts=5, raw_samples=128)
        u_new = cand.detach().numpy().ravel()
        y_new = oracle_observe(oracle, u_new, "pilot", rng)
        U = np.vstack([U, u_new]); y = np.append(y, y_new)
        cost += COST["pilot"]; best = max(best, y_new)
        curve.append((float(cost), float(best)))

    Xt = torch.tensor(U, **TKW); Yt = torch.tensor(y.reshape(-1, 1), **TKW)
    model = SingleTaskGP(Xt, Yt, outcome_transform=Standardize(m=1))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    rec = recommend_singletask(model)
    rec_true = float(oracle_mean(oracle, rec.reshape(1, -1), 1.0)[0])
    return {"cost": cost, "n_pilot": len(y), "best_pilot_obs": float(best),
            "curve": curve, "rec": rec, "rec_true": rec_true}


# --------------------------------------------------------------------------- #
#  Plotting                                                                    #
# --------------------------------------------------------------------------- #
def interp_curve(curve, grid):
    c = np.array([p[0] for p in curve]); v = np.array([p[1] for p in curve])
    v = np.where(np.isfinite(v), v, np.nan)
    return np.interp(grid, c, v, left=np.nan, right=v[-1])


#  The two live arms carry the comparison, so they get the two hues; the biased
#  replay is drawn as a recessive grey reference, like the oracle ceiling.
C_LIVE, C_PILOT, C_REPLAY = "#4C9BE8", "#D64550", "0.55"


def plot(live_runs, po_runs, ms_runs, oracle_best, out_path, budget):
    grid = np.linspace(0, budget, 200)
    series = [
        (np.vstack([interp_curve(r["curve"], grid) for r in live_runs]),
         C_LIVE, "-", "multi-scale BO (live search)"),
        (np.vstack([interp_curve(r["curve"], grid) for r in po_runs]),
         C_PILOT, "-", "pilot-only BO (live search)"),
        (np.vstack([interp_curve(r["curve"], grid) for r in ms_runs]),
         C_REPLAY, ":", "campaign replay (biased reference)"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2),
                                   gridspec_kw={"width_ratios": [1.6, 1]})

    for arr, col, ls, lab in series:
        m = np.nanmean(arr, axis=0); s = np.nanstd(arr, axis=0)
        ax1.plot(grid, m, color=col, lw=2, ls=ls, label=lab)
        if ls == "-":
            ax1.fill_between(grid, m - s, m + s, color=col, alpha=0.15, lw=0)
    ax1.axhline(oracle_best, ls="--", color="0.4", lw=1.3,
                label="oracle pilot optimum")
    ax1.set_xlabel("cumulative cost (EUR)")
    ax1.set_ylabel("best pilot yield observed")
    ax1.set_title("Best pilot yield vs cost (mean +/- std over seeds)")
    ax1.legend(fontsize=8, loc="lower right"); ax1.grid(True, alpha=0.2)

    # recommendation quality (true oracle yield at recommended recipe)
    recs = [np.array([r["rec_true"] for r in runs])
            for runs in (live_runs, po_runs, ms_runs)]
    ax2.bar(range(3), [r.mean() for r in recs], yerr=[r.std() for r in recs],
            capsize=4, width=0.6, color=[C_LIVE, C_PILOT, C_REPLAY])
    ax2.axhline(oracle_best, ls="--", color="0.4", lw=1.3)
    ax2.text(2.45, oracle_best, "oracle optimum", va="bottom", ha="right",
             fontsize=8, color="0.35")
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(["multi-scale\n(live)", "pilot-only\n(live)",
                         "replay\n(biased)"])
    ax2.set_ylabel("true pilot yield at recommendation")
    ax2.set_title("Recommendation quality at equal cost")
    ax2.grid(True, axis="y", alpha=0.2)

    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)


# --------------------------------------------------------------------------- #
model_noise: dict = {}   # set per-process, read by oracle_observe


def _setup(df: pd.DataFrame):
    """Build the frozen oracle. Seeded so every worker gets the SAME oracle --
    the arms must be scored against one identical ground truth.

    Threads are pinned to 1 as part of that guarantee, not just for speed: the
    thread count changes float reduction order inside the GP fit, which moves
    the fitted hyperparameters slightly. Without this the parent process (which
    computes the reference ceiling) builds a marginally different oracle from
    the workers (which score the arms).
    """
    global model_noise
    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    oracle, model_noise = build_oracle(df)
    return oracle


def _run_seed(args_tuple):
    """One seed of all three arms. Top-level so it survives pickling."""
    seed, budget, max_iter, fantasies, restarts, raw = args_tuple
    torch.set_num_threads(1)          # one core each; seeds run in parallel
    df = pd.read_csv(REF)
    df.columns = [c.strip() for c in df.columns]
    oracle = _setup(df)

    live = run_multiscale_live(oracle, budget, np.random.default_rng(1000 + seed),
                               max_iter=max_iter, fantasies=fantasies,
                               restarts=restarts, raw=raw)
    po = run_pilot_only(oracle, budget, np.random.default_rng(1000 + seed))
    ms = run_multiscale(oracle, df, np.random.default_rng(1000 + seed))
    return seed, live, po, ms


def main() -> None:
    global model_noise
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--budget", type=float, default=None,
                    help="EUR; default = the replayed campaign's total cost")
    ap.add_argument("--jobs", type=int, default=0,
                    help="parallel seeds (0 = min(seeds, cpu_count-1))")
    ap.add_argument("--max-iter", type=int, default=250,
                    help="cap on live multi-scale BO iterations per seed")
    ap.add_argument("--fantasies", type=int, default=4)
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--raw", type=int, default=16)
    args = ap.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True); os.makedirs(TAB_DIR, exist_ok=True)
    df = pd.read_csv(REF); df.columns = [c.strip() for c in df.columns]

    print("building frozen oracle (SingleTaskMultiFidelityGP on all data)...")
    oracle = _setup(df)
    print(f"  per-scale observation noise: "
          f"{ {k: round(v, 3) for k, v in model_noise.items()} }")

    # global optimum of the oracle's pilot surface, for reference (found by
    # optimizing the oracle's own posterior mean, so no arm can exceed it)
    rec_star = recommend_multifidelity(oracle)
    oracle_best = float(oracle_mean(oracle, rec_star.reshape(1, -1), 1.0)[0])

    campaign_cost = float(df["cumulative_cost_eur"].max())
    budget = args.budget if args.budget else campaign_cost
    print(f"campaign total cost = {campaign_cost:.0f} EUR -> budget = "
          f"{budget:.0f} EUR (pilot-only can afford "
          f"{int(budget // COST['pilot'])} pilot runs)")
    if abs(budget - campaign_cost) > 1e-6:
        # The replay arm always costs exactly what the campaign cost -- it is a
        # fixed recording, not a budgeted search -- so it drops out of the
        # equal-cost comparison as soon as the budget is overridden.
        print(f"  NOTE: --budget overrides the campaign cost, so the replay arm "
              f"(fixed at\n  {campaign_cost:.0f} EUR) is NOT equal-cost with the "
              f"live arms here; ignore its column.")

    jobs = args.jobs or max(1, min(args.seeds, (os.cpu_count() or 2) - 1))
    print(f"running {args.seeds} seeds x 3 arms on {jobs} process(es); the live "
          f"multi-scale arm is slow (MFKG)...\n")

    payload = [(s, budget, args.max_iter, args.fantasies, args.restarts,
                args.raw) for s in range(args.seeds)]
    results = {}
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for seed, live, po, ms in ex.map(_run_seed, payload):
            results[seed] = (live, po, ms)
            print(f"seed {seed}: live multi-scale rec_true={live['rec_true']:.3f} "
                  f"({live['n_iter']} iters, {live['cost']:.0f} EUR, "
                  f"{live['n_by_scale']}) | pilot-only "
                  f"rec_true={po['rec_true']:.3f} ({po['n_pilot']} pilots) | "
                  f"replay rec_true={ms['rec_true']:.3f}")

    live_runs = [results[s][0] for s in sorted(results)]
    po_runs = [results[s][1] for s in sorted(results)]
    ms_runs = [results[s][2] for s in sorted(results)]

    def arr(runs, key):
        return np.array([r[key] for r in runs])

    print("\n=== equal-cost comparison "
          f"({args.seeds} seeds, budget {budget:.0f} EUR) ===")
    summ = pd.DataFrame({
        "metric": ["recommendation true pilot Y", "best observed pilot Y",
                   "total cost spent (EUR)"],
        "multiscale_live_mean": [arr(live_runs, "rec_true").mean(),
                                 arr(live_runs, "best_pilot_obs").mean(),
                                 arr(live_runs, "cost").mean()],
        "multiscale_live_std": [arr(live_runs, "rec_true").std(),
                                arr(live_runs, "best_pilot_obs").std(),
                                arr(live_runs, "cost").std()],
        "pilot_only_mean": [arr(po_runs, "rec_true").mean(),
                            arr(po_runs, "best_pilot_obs").mean(),
                            arr(po_runs, "cost").mean()],
        "pilot_only_std": [arr(po_runs, "rec_true").std(),
                           arr(po_runs, "best_pilot_obs").std(),
                           arr(po_runs, "cost").std()],
        "replay_biased_mean": [arr(ms_runs, "rec_true").mean(),
                               arr(ms_runs, "best_pilot_obs").mean(),
                               arr(ms_runs, "cost").mean()],
        "replay_biased_std": [arr(ms_runs, "rec_true").std(),
                              arr(ms_runs, "best_pilot_obs").std(),
                              arr(ms_runs, "cost").std()],
    })
    print(summ.round(3).to_string(index=False))
    print(f"\noracle pilot optimum (reference ceiling): {oracle_best:.3f}")

    # Equal-cost is the premise of the whole comparison, so assert it rather
    # than trust it: both live arms must stay within budget.
    over = [(s, c) for s, c in zip(sorted(results), arr(live_runs, "cost"))
            if c > budget + 1e-6]
    if over:
        print(f"\n*** WARNING: live multi-scale exceeded the {budget:.0f} EUR "
              f"budget on seeds {[s for s, _ in over]} "
              f"(max {max(c for _, c in over):.0f} EUR) -- NOT equal-cost. ***")
    else:
        print(f"budget check: both live arms stayed within {budget:.0f} EUR "
              f"(multi-scale spent {arr(live_runs, 'cost').mean():.0f} avg, "
              f"pilot-only {arr(po_runs, 'cost').mean():.0f}; leftovers are "
              f"granularity).")

    live_rec, po_rec = arr(live_runs, "rec_true"), arr(po_runs, "rec_true")
    delta = live_rec.mean() - po_rec.mean()

    # Paired over seeds: both live arms see the same oracle and the same seed,
    # so the per-seed difference removes the (large) between-seed variance.
    paired = live_rec - po_rec
    se = paired.std(ddof=1) / np.sqrt(len(paired)) if len(paired) > 1 else 0.0
    wins = int((paired > 0).sum())

    # The sign of `delta` alone is not a verdict: with this few seeds and this
    # much between-seed variance the interval routinely straddles zero, and a
    # bare "HELPS"/"HURTS" read off the sign would be noise reported as a
    # finding. Only claim a direction if the interval excludes 0.
    print("\n=== VERDICT (fair, live-vs-live) ===")
    ci_lo, ci_hi = delta - 1.96 * se, delta + 1.96 * se
    print(f"multi-scale (live) - pilot-only (live) = {delta:+.3f} "
          f"(paired SE {se:.3f}, 95% CI [{ci_lo:+.2f}, {ci_hi:+.2f}], "
          f"wins {wins}/{len(paired)})")
    if ci_lo > 0:
        print("-> at equal cost, cheap-scale information HELPS "
              "(CI excludes 0).")
    elif ci_hi < 0:
        print("-> at equal cost, cheap-scale information HURTS "
              "(CI excludes 0).")
    else:
        print("-> NO DETECTABLE DIFFERENCE: the CI straddles 0, so these seeds "
              "cannot\n   separate the two strategies in either direction. "
              "Do not read the sign of\n   the mean as a result; it is within "
              "noise.")
        need = int(np.ceil((1.96 * paired.std(ddof=1) / max(abs(delta), 1e-9))
                           ** 2)) if abs(delta) > 1e-9 else None
        if need and need < 10000:
            print(f"   (Resolving an effect this size at 95% would take "
                  f"~{need} seeds.)")

    print("\nFor reference, the biased campaign replay scores "
          f"{arr(ms_runs, 'rec_true').mean():.3f} "
          f"({arr(ms_runs, 'rec_true').mean() - po_rec.mean():+.3f} vs "
          "pilot-only).\nThat arm does not search -- it re-runs the finished "
          "campaign's own recipes, and\nthe oracle is fitted to that same "
          "campaign, so it is an UPPER BOUND, not a result.")

    summ.to_csv(os.path.join(TAB_DIR, "task3_equal_cost_summary.csv"),
                index=False)
    per_seed = pd.DataFrame({
        "seed": sorted(results),
        "multiscale_live_rec_true": live_rec,
        "pilot_only_rec_true": po_rec,
        "replay_rec_true": arr(ms_runs, "rec_true"),
        "paired_diff_live_minus_pilot": paired,
        "multiscale_live_cost": arr(live_runs, "cost"),
        "multiscale_live_iters": arr(live_runs, "n_iter"),
        # The live arm's recommendation is read off the pilot slice of a GP fit
        # on per-fidelity min-max scaled Y (mfbo_v1's FidelityScaler). With only
        # 2-3 pilot runs that slice is degenerate -- min-max maps 2 points to
        # exactly {0, 1} regardless of their values -- so the recommendation is
        # driven by micro structure alone. Record the scale mix: it is the first
        # thing to check when a seed's rec_true collapses.
        "multiscale_live_n_pilot": [r["n_by_scale"]["pilot"] for r in live_runs],
        "multiscale_live_n_bench": [r["n_by_scale"]["bench"] for r in live_runs],
        "multiscale_live_n_micro": [r["n_by_scale"]["micro"] for r in live_runs],
    })
    per_seed.to_csv(os.path.join(TAB_DIR, "task3_per_seed.csv"), index=False)

    out = os.path.join(FIG_DIR, "task3_pilot_vs_multiscale.png")
    plot(live_runs, po_runs, ms_runs, oracle_best, out, budget)
    print(f"\nsaved: {out}\n       "
          f"{os.path.join(TAB_DIR, 'task3_equal_cost_summary.csv')}\n       "
          f"{os.path.join(TAB_DIR, 'task3_per_seed.csv')}")


if __name__ == "__main__":
    main()
