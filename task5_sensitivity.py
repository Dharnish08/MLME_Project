"""Additional task 5: global sensitivity analysis of the fitted surrogate.

We refit the project's surrogate (``SingleTaskMultiFidelityGP``) on *all* logged
experiments, then treat its posterior mean at the pilot fidelity as a cheap,
smooth response surface

        Y_hat = g(T, pH, F1, F2, F3),   inputs normalised to [0, 1]^5.

A variance-based (Sobol) global sensitivity analysis on g answers:

  * first-order index  S1_i  -> how much of the variance in product amount is
    explained by variable i *on its own*;
  * total-order index  ST_i  -> variable i's total contribution, including
    every interaction it takes part in;
  * second-order index S2_ij -> the variance explained by the *pair* (i, j)
    beyond their individual main effects (i.e. genuine interaction).

ST_i - S1_i is the share of i's influence that only appears through
interactions.  This is a purely offline analysis (no API calls): the GP is the
oracle, evaluated tens of thousands of times by the Saltelli sampler.

    python task5_sensitivity.py
"""
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
from botorch import fit_gpytorch_mll
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as sobol_sample

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "results", "tables_initial_experiments",
                   "cleaned_full_optimization_log_reference.csv")
FIG_DIR = os.path.join(HERE, "results", "figures_additional_tasks")
TAB_DIR = os.path.join(HERE, "results", "tables_additional_tasks")

VARS = ["T", "pH", "F1", "F2", "F3"]
NORM_COLS = ["T_norm", "pH_norm", "F1_norm", "F2_norm", "F3_norm"]
TKW = {"dtype": torch.double, "device": torch.device("cpu")}

N_SOBOL = 1024          # base Saltelli sample size (power of 2)
TARGET_FIDELITY = 1.0   # pilot
# SALib seeds its own sampler and its bootstrap CIs independently of numpy's
# global seed; without passing this explicitly the indices move between runs.
SEED = 0


def design_collinearity(df: pd.DataFrame, thresh: float = 0.4) -> pd.DataFrame:
    """Input pairs that are correlated in the *sampled design*.

    Sobol indices are defined for independent, uniformly distributed inputs.
    These experiments were chosen by a BO loop, not by a space-filling design,
    so the inputs are not independent.  Where two inputs are strongly
    correlated in the design, the GP cannot tell their effects apart -- ARD
    will hand the shared variance to whichever one it fits with the shorter
    lengthscale, and Sobol then reports that variable as dominant and the other
    as inert.  That split is an artefact of the design, not a property of the
    process, so it is reported rather than hidden.
    """
    C = df[NORM_COLS].corr(method="spearman")
    rows = []
    for i in range(len(VARS)):
        for j in range(i + 1, len(VARS)):
            r = C.iloc[i, j]
            if abs(r) >= thresh:
                rows.append({"pair": f"{VARS[i]}-{VARS[j]}", "spearman_rho": r})
    return pd.DataFrame(rows).sort_values(
        "spearman_rho", key=abs, ascending=False) if rows else pd.DataFrame(
        columns=["pair", "spearman_rho"])


def target_fidelity_support(df: pd.DataFrame) -> pd.DataFrame:
    """How much the design actually varies each input *at the target fidelity*.

    The indices below describe g at the pilot slice, but the GP only knows that
    slice through the pilot runs plus whatever the fidelity kernel carries over
    from micro/bench. If an input barely moves across the pilot runs, the pilot
    surface's dependence on it is inherited by extrapolation, not measured -- so
    its index is a statement about the cheap scales, not about the pilot.
    """
    p = df[df["Fidelity_Scaled"] == TARGET_FIDELITY]
    return pd.DataFrame({
        "variable": VARS,
        "n_target_runs": len(p),
        "design_std_at_target": [p[c].std() if len(p) > 1 else np.nan
                                 for c in NORM_COLS],
    })


def fit_surrogate() -> SingleTaskMultiFidelityGP:
    df = pd.read_csv(REF)
    df.columns = [c.strip() for c in df.columns]
    X = torch.tensor(df[NORM_COLS].values, **TKW)
    f = torch.tensor(df["Fidelity_Scaled"].values, **TKW).unsqueeze(-1)
    Y = torch.tensor(df["Y"].values, **TKW).unsqueeze(-1)
    X6 = torch.cat([X, f], dim=1)
    model = SingleTaskMultiFidelityGP(
        X6, Y, outcome_transform=Standardize(m=1), data_fidelities=[5])
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval()
    print(f"surrogate fitted on {len(df)} experiments "
          f"({df['scale'].value_counts().to_dict()})")
    return model


def surrogate_mean(model, U: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """Posterior mean of g at recipes U (shape [n, 5]) at the pilot fidelity."""
    means = []
    fid = np.full((U.shape[0], 1), TARGET_FIDELITY)
    X6 = np.hstack([U, fid])
    with torch.no_grad():
        for i in range(0, X6.shape[0], chunk):
            xb = torch.tensor(X6[i:i + chunk], **TKW)
            means.append(model.posterior(xb).mean.squeeze(-1).cpu().numpy())
    return np.concatenate(means)


def run_sobol(model) -> dict:
    problem = {"num_vars": 5, "names": VARS,
               "bounds": [[0.0, 1.0]] * 5}
    param = sobol_sample.sample(problem, N_SOBOL, calc_second_order=True,
                                seed=SEED)
    y = surrogate_mean(model, param)
    Si = sobol_analyze.analyze(problem, y, calc_second_order=True,
                               print_to_console=False, seed=SEED)
    print(f"evaluated surrogate at {len(param)} Saltelli points")
    return problem, Si


def group_labels(coll: pd.DataFrame) -> list:
    """Merge collinear variables into groups (connected components).

    Individual indices are not identifiable inside a collinear cluster, but the
    *group's* index is: it measures what the whole cluster explains, which does
    not depend on how the shared variance is split within it.
    """
    parent = {v: v for v in VARS}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for p in coll["pair"]:
        a, b = p.split("-")
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    members = {}
    for v in VARS:
        members.setdefault(find(v), []).append(v)
    name_of = {root: "+".join(m) if len(m) > 1 else m[0]
               for root, m in members.items()}
    return [name_of[find(v)] for v in VARS]


def run_sobol_grouped(model, groups: list) -> pd.DataFrame:
    """Sobol indices over variable groups rather than single variables."""
    problem = {"num_vars": 5, "names": VARS, "bounds": [[0.0, 1.0]] * 5,
               "groups": groups}
    param = sobol_sample.sample(problem, N_SOBOL, calc_second_order=False,
                                seed=SEED)
    y = surrogate_mean(model, param)
    Si = sobol_analyze.analyze(problem, y, calc_second_order=False,
                               print_to_console=False, seed=SEED)
    uniq = list(dict.fromkeys(groups))
    return pd.DataFrame({"group": uniq, "S1": Si["S1"], "S1_conf": Si["S1_conf"],
                         "ST": Si["ST"], "ST_conf": Si["ST_conf"]}
                        ).sort_values("ST", ascending=False)


def plot_indices(Si: dict, out_path: str, coll: pd.DataFrame = None) -> None:
    x = np.arange(len(VARS))
    w = 0.4
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.4))

    # ---- first-order vs total-order ----
    ax1.bar(x - w / 2, Si["S1"], w, yerr=Si["S1_conf"], capsize=3,
            color="#4C9BE8", label="S1 (main effect)")
    ax1.bar(x + w / 2, Si["ST"], w, yerr=Si["ST_conf"], capsize=3,
            color="#D64550", label="ST (total effect)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(VARS)
    ax1.set_ylabel("Sobol index (fraction of Y variance)")
    ax1.set_title("Main vs total effect on product amount")
    ax1.axhline(0, color="0.6", lw=0.8)
    ax1.legend()
    ax1.grid(True, axis="y", alpha=0.25)

    # ---- second-order interaction heatmap ----
    S2 = np.array(Si["S2"], dtype=float)
    M = np.full((5, 5), np.nan)
    for i in range(5):
        for j in range(i + 1, 5):
            M[i, j] = S2[i, j]
            M[j, i] = S2[i, j]
    # Sequential magnitude -> a single hue, light to dark (a multi-hue ramp
    # invents structure that isn't in the numbers).
    im = ax2.imshow(M, cmap="Blues", vmin=0)
    ax2.set_xticks(x); ax2.set_xticklabels(VARS)
    ax2.set_yticks(x); ax2.set_yticklabels(VARS)
    ax2.set_title("Second-order interaction indices $S_{ij}$")
    hi = np.nanmax(M)
    for i in range(5):
        for j in range(5):
            if not np.isnan(M[i, j]):
                ax2.text(j, i, f"{M[i, j]:.02f}", ha="center", va="center",
                         color="white" if M[i, j] > hi * 0.6 else "0.15",
                         fontsize=9)
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    # A reader of the figure alone must not walk away with "F3 is the lever":
    # the per-variable split inside a collinear cluster is not identifiable.
    if coll is not None and len(coll):
        pairs = ", ".join(f"{r.pair} (rho={r.spearman_rho:+.2f})"
                          for r in coll.itertuples())
        fig.text(0.5, 0.015,
                 f"Caveat: Sobol assumes independent inputs, but this "
                 f"BO-selected design is collinear: {pairs}. Indices within a "
                 f"collinear pair are not identifiable\n(the GP hands their "
                 f"shared variance to one of them); read those variables "
                 f"jointly. See task5_sobol_grouped.csv.",
                 ha="center", va="bottom", fontsize=8, color="0.25")
        fig.tight_layout(rect=[0, 0.085, 1, 1])
    else:
        fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TAB_DIR, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)

    model = fit_surrogate()
    problem, Si = run_sobol(model)

    ref = pd.read_csv(REF)
    ref.columns = [c.strip() for c in ref.columns]
    coll = design_collinearity(ref)

    # ---- first/total-order table ----
    first = pd.DataFrame({
        "variable": VARS,
        "S1": Si["S1"], "S1_conf": Si["S1_conf"],
        "ST": Si["ST"], "ST_conf": Si["ST_conf"],
        "interaction_share_ST_minus_S1": np.array(Si["ST"]) - np.array(Si["S1"]),
    })
    print("\n=== first-order (S1) and total-order (ST) indices ===")
    print(first.round(3).to_string(index=False))

    # ---- pairwise second-order table ----
    S2 = np.array(Si["S2"], dtype=float)
    pairs = []
    for i in range(5):
        for j in range(i + 1, 5):
            pairs.append({"pair": f"{VARS[i]}*{VARS[j]}",
                          "S2": S2[i, j], "S2_conf": Si["S2_conf"][i, j]})
    second = pd.DataFrame(pairs).sort_values("S2", ascending=False)
    print("\n=== second-order interaction indices (S2), largest first ===")
    print(second.round(3).to_string(index=False))

    # ---- interpretation ----
    order = first.sort_values("ST", ascending=False)["variable"].tolist()
    top = order[0]
    strong_pair = second.iloc[0]
    print("\n=== interpretation ===")
    print(f"* Ranking of drivers by total effect: {' > '.join(order)}.")
    print(f"* '{top}' dominates the product amount.")
    # A low ST is only evidence of inertness for a variable whose effect is
    # identifiable. Inside a collinear cluster the GP hands the shared variance
    # to one member, so the *other* member's ST is driven to ~0 by the design,
    # not by the process -- calling it inert would contradict the collinearity
    # caveat printed below. Exclude clustered variables from the claim.
    clustered = {v for p in coll["pair"] for v in p.split("-")} if len(coll) \
        else set()
    weak = first.loc[first["ST"] < 0.05, "variable"].tolist()
    inert = [v for v in weak if v not in clustered]
    suppressed = [v for v in weak if v in clustered]
    if inert:
        print(f"* {', '.join(inert)} are near-inert (ST < 0.05): the surrogate "
              f"sees little product response to them over their range.")
    if suppressed:
        verb = "scores" if len(suppressed) == 1 else "score"
        print(f"* {', '.join(suppressed)} also {verb} ST < 0.05, but is "
              f"collinear with a stronger\n  variable, so this low ST is NOT "
              f"evidence of inertness -- the design, not the\n  process, drove "
              f"it there. Do not report it as unimportant.")
    # An S2 point estimate above an arbitrary cutoff is not evidence of an
    # interaction: SALib's bootstrap CI here is routinely wider than the
    # estimate itself, so a fixed threshold turns noise into a recommendation
    # to "tune these jointly". Claim an interaction only if its CI excludes 0.
    sig = second[(second["S2"] - second["S2_conf"]) > 0]
    if len(sig):
        sp = sig.iloc[0]
        print(f"* Strongest interaction: {sp['pair']} (S2={sp['S2']:.3f} "
              f"+/-{sp['S2_conf']:.3f}, CI excludes 0) -- these two variables "
              f"must be tuned jointly, not one at a time.")
    else:
        print(f"* NO second-order interaction is distinguishable from zero: "
              f"every pair's\n  bootstrap CI includes 0 (largest is "
              f"{strong_pair['pair']}, S2={strong_pair['S2']:.3f} "
              f"+/-{strong_pair['S2_conf']:.3f}).\n  The response is close to "
              f"additive over the region the surrogate trusts, so\n  variables "
              f"can largely be tuned independently -- but note the CIs are as\n"
              f"  wide as the estimates, so this rules interactions IN or OUT "
              f"only weakly.")

    # ---- validity caveat: how much pilot data backs the pilot slice? ----
    sup = target_fidelity_support(ref)
    n_target = int(sup["n_target_runs"].iloc[0])
    print("\n=== caveat: how much pilot data actually backs this surface? ===")
    print(f"The indices describe the surrogate at the PILOT fidelity, but only "
          f"{n_target} of the\n{len(ref)} logged runs are pilot runs. Spread of "
          f"each input across those runs:")
    print(sup[["variable", "design_std_at_target"]].round(3).to_string(
        index=False))
    thin = sup.loc[sup["design_std_at_target"] < 0.15, "variable"].tolist()
    if thin:
        print(f"\n{', '.join(thin)} barely vary across the pilot runs, so the "
              f"pilot surface's\ndependence on them is carried over from the "
              f"micro/bench data by the fidelity\nkernel -- extrapolated, not "
              f"measured. Treat the whole ranking as 'what the cheap\nscales "
              f"say, transported to pilot', and confirm it with pilot runs that "
              f"actually\nvary these inputs before acting on it.")

    print("\n=== caveat: Sobol assumes independent inputs ===")
    if len(coll):
        print("These inputs are correlated in the BO-selected design, so the "
              "surrogate\ncannot separate their effects and the S1/ST split "
              "between them is NOT identifiable:")
        print(coll.round(3).to_string(index=False))
        flagged = {v for p in coll["pair"] for v in p.split("-")}
        if top in flagged:
            partners = sorted({v for p in coll["pair"] if top in p.split("-")
                               for v in p.split("-") if v != top})
            print(f"\n'{top}' is reported as the dominant driver, but it is "
                  f"collinear with {', '.join(partners)} here.\nRead this as "
                  f"'{top} and {'/'.join(partners)} jointly drive the yield' -- "
                  f"confirming that\n{top} specifically is the lever needs runs "
                  f"that vary {top} while holding {'/'.join(partners)} fixed.")

        # The grouped index IS identifiable under collinearity: merge each
        # collinear cluster and ask what the cluster as a whole explains.
        groups = group_labels(coll)
        grouped = run_sobol_grouped(model, groups)
        print("\n=== grouped indices (collinear variables merged) ===")
        print("These ARE identifiable -- the split within a group is not:")
        print(grouped.round(3).to_string(index=False))
        grouped.to_csv(os.path.join(TAB_DIR, "task5_sobol_grouped.csv"),
                       index=False)
    else:
        print("No input pair exceeds |rho| = 0.4 in the design; the indices "
              "are safe to read per-variable.")

    first.to_csv(os.path.join(TAB_DIR, "task5_sobol_first_total.csv"),
                 index=False)
    second.to_csv(os.path.join(TAB_DIR, "task5_sobol_second_order.csv"),
                  index=False)
    fig_out = os.path.join(FIG_DIR, "task5_sensitivity.png")
    plot_indices(Si, fig_out, coll)
    print(f"\nsaved: {fig_out}")
    print(f"       {os.path.join(TAB_DIR, 'task5_sobol_first_total.csv')}")
    print(f"       {os.path.join(TAB_DIR, 'task5_sobol_second_order.csv')}")


if __name__ == "__main__":
    main()
