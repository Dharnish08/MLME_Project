"""Additional task 5: global sensitivity analysis of the fitted surrogate.

We refit the project's surrogate (``SingleTaskMultiFidelityGP``) on *all* logged
experiments and run a variance-based (Sobol) analysis on its posterior mean

        Y_hat = g(T, pH, F1, F2, F3),   inputs normalised to [0, 1]^5.

  * first-order index  S1_i  -> variance in product amount explained by i alone;
  * total-order index  ST_i  -> i's total contribution, incl. every interaction;
  * second-order index S2_ij -> variance explained by the *pair* beyond their
    main effects (i.e. genuine interaction).

WHY THIS SCRIPT DOES NOT JUST REPORT THE PILOT SLICE
----------------------------------------------------
An earlier version of this analysis ran Sobol only at the pilot fidelity and
concluded "F3 dominates, T is inert, the response is additive".  All three
claims were artefacts.  Only 3 of the 187 logged runs are pilot runs, and the
BO loop had already converged, so at pilot those 3 runs span T = 34.6-40 C (of
20-60) and pH = 8.07-8.6 (of 3-9.5) while F3 swings nearly full range.  A Sobol
analysis of that slice is a description of three points: it *must* hand ~all
variance to F3, and it reports T's index as 0.000 because the design never
moved T -- not because T does nothing.  The optimizer converged onto the T/pH
optimum and in doing so destroyed the variance the GSA needed to see it.

So this script now:
  1. runs the GSA at EVERY fidelity, not just pilot, and reports them together;
  2. gates each index on ``design_support`` -- if an input barely varies at that
     fidelity, its index is reported as NOT IDENTIFIABLE rather than as a
     finding;
  3. cross-checks the surrogate against a non-parametric sweep of the RAW data,
     which uses no model and no monotonicity assumption.  T's effect is a bump
     peaking near 37 C, so rank correlation (rho = -0.12) misses it entirely;
     binned means do not.

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

# Real-unit ranges, for reporting design coverage in units a process engineer
# can act on (Table 1 of the project brief).
REAL_RANGE = {"T": (20.0, 60.0), "pH": (3.0, 9.5),
              "F1": (0.0, 2.0), "F2": (0.0, 2.0), "F3": (0.0, 2.0)}

FIDELITY_OF = {"micro": 0.0, "bench": 0.5, "pilot": 1.0}

N_SOBOL = 1024          # base Saltelli sample size (power of 2)
TARGET_FIDELITY = 1.0   # pilot -- the scale we ultimately optimise for
# SALib seeds its own sampler and its bootstrap CIs independently of numpy's
# global seed; without passing this explicitly the indices move between runs.
SEED = 0

# A uniform design on [0,1] has std = 0.289. Below this an input is not
# meaningfully exercised, so any index attributed to it is extrapolation.
MIN_DESIGN_STD = 0.15
# Raw-data swing (max/min of binned mean Y) above which we call an effect real.
RAW_FOLD_ALERT = 2.0
# ST below which the surrogate is calling a variable inert.
INERT_ST = 0.05

# Categorical palette slots 1/2/3, fixed order (dataviz reference palette).
COLOR_OF = {"micro": "#2a78d6", "bench": "#008300", "pilot": "#e87ba4"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SURFACE = "#fcfcfb"


# --------------------------------------------------------------------------
# data / model
# --------------------------------------------------------------------------
def load_ref() -> pd.DataFrame:
    df = pd.read_csv(REF)
    df.columns = [c.strip() for c in df.columns]
    return df


def fit_surrogate(df: pd.DataFrame) -> SingleTaskMultiFidelityGP:
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


def surrogate_mean(model, U: np.ndarray, fidelity: float,
                   chunk: int = 4096) -> np.ndarray:
    """Posterior mean of g at recipes U (shape [n, 5]) at a given fidelity."""
    means = []
    X6 = np.hstack([U, np.full((U.shape[0], 1), fidelity)])
    with torch.no_grad():
        for i in range(0, X6.shape[0], chunk):
            xb = torch.tensor(X6[i:i + chunk], **TKW)
            means.append(model.posterior(xb).mean.squeeze(-1).cpu().numpy())
    return np.concatenate(means)


# --------------------------------------------------------------------------
# design diagnostics -- what the data can and cannot support
# --------------------------------------------------------------------------
def design_support(df: pd.DataFrame) -> pd.DataFrame:
    """Per-scale spread of each input in the *sampled design*.

    Sobol describes g over the full unit cube, but the GP only knows a slice
    through the runs made at that scale (plus whatever the fidelity kernel
    carries across). Where an input barely moves, the slice's dependence on it
    is inherited by extrapolation, not measured.
    """
    rows = []
    for scale, sub in df.groupby("scale"):
        for v, c in zip(VARS, NORM_COLS):
            lo, hi = REAL_RANGE[v]
            rows.append({
                "scale": scale,
                "variable": v,
                "n_runs": len(sub),
                "design_std": sub[c].std() if len(sub) > 1 else np.nan,
                "real_min": lo + sub[c].min() * (hi - lo),
                "real_max": lo + sub[c].max() * (hi - lo),
                "range_covered_frac": (sub[c].max() - sub[c].min())
                if len(sub) > 1 else np.nan,
            })
    out = pd.DataFrame(rows)
    out["identifiable"] = out["design_std"] >= MIN_DESIGN_STD
    return out


def raw_effect_screen(df: pd.DataFrame, scale: str = "micro",
                      nbins: int = 6) -> pd.DataFrame:
    """Model-free effect screen on the RAW data at one scale.

    Deliberately assumes nothing: no GP, no monotonicity. Bins each input into
    quantiles and compares mean Y across bins. This is the check that catches a
    unimodal driver -- T peaks near 37 C, so Spearman's rho is ~0 while the
    binned means swing several-fold. Reported alongside rho so the contrast is
    visible.
    """
    s = df[df["scale"] == scale]
    rows = []
    for v, c in zip(VARS, NORM_COLS):
        if len(s) < nbins * 2:
            continue
        try:
            bins = pd.qcut(s[c], nbins, duplicates="drop")
        except ValueError:
            continue
        g = s.groupby(bins, observed=True)["Y"].mean()
        if len(g) < 2:
            continue
        lo, hi = float(g.min()), float(g.max())
        rows.append({
            "scale": scale, "variable": v, "n": len(s),
            "min_bin_meanY": lo, "max_bin_meanY": hi,
            "swing_abs": hi - lo,
            "fold_change": (hi / lo) if lo > 1e-9 else np.inf,
            "spearman_rho": s[c].corr(s["Y"], method="spearman"),
        })
    return pd.DataFrame(rows).sort_values("fold_change", ascending=False)


def design_collinearity(df: pd.DataFrame, thresh: float = 0.4) -> pd.DataFrame:
    """Input pairs correlated in the *sampled design*.

    Sobol indices are defined for independent inputs. These experiments were
    chosen by a BO loop, not a space-filling design, so where two inputs are
    strongly correlated the GP cannot tell their effects apart -- ARD hands the
    shared variance to one of them and Sobol then reports that one as dominant
    and the other as inert. That split is an artefact of the design.
    """
    C = df[NORM_COLS].corr(method="spearman")
    rows = []
    for i in range(len(VARS)):
        for j in range(i + 1, len(VARS)):
            r = C.iloc[i, j]
            if abs(r) >= thresh:
                rows.append({"pair": f"{VARS[i]}-{VARS[j]}", "spearman_rho": r})
    return (pd.DataFrame(rows).sort_values("spearman_rho", key=abs,
                                           ascending=False)
            if rows else pd.DataFrame(columns=["pair", "spearman_rho"]))


# --------------------------------------------------------------------------
# Sobol
# --------------------------------------------------------------------------
def run_sobol_at(model, fidelity: float) -> tuple:
    problem = {"num_vars": 5, "names": VARS, "bounds": [[0.0, 1.0]] * 5}
    param = sobol_sample.sample(problem, N_SOBOL, calc_second_order=True,
                                seed=SEED)
    y = surrogate_mean(model, param, fidelity)
    Si = sobol_analyze.analyze(problem, y, calc_second_order=True,
                               print_to_console=False, seed=SEED)
    return problem, Si, len(param)


def indices_frame(Si: dict, scale: str, support: pd.DataFrame) -> pd.DataFrame:
    sup = support[support["scale"] == scale].set_index("variable")
    out = pd.DataFrame({
        "scale": scale,
        "variable": VARS,
        "S1": Si["S1"], "S1_conf": Si["S1_conf"],
        "ST": Si["ST"], "ST_conf": Si["ST_conf"],
        "interaction_share_ST_minus_S1": np.array(Si["ST"]) - np.array(Si["S1"]),
    })
    out["design_std"] = [sup.loc[v, "design_std"] if v in sup.index else np.nan
                         for v in VARS]
    out["identifiable"] = out["design_std"] >= MIN_DESIGN_STD
    return out


def second_order_frame(Si: dict, scale: str) -> pd.DataFrame:
    S2 = np.array(Si["S2"], dtype=float)
    rows = []
    for i in range(5):
        for j in range(i + 1, 5):
            rows.append({"scale": scale, "pair": f"{VARS[i]}*{VARS[j]}",
                         "S2": S2[i, j], "S2_conf": Si["S2_conf"][i, j]})
    return pd.DataFrame(rows).sort_values("S2", ascending=False)


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


def run_sobol_grouped(model, groups: list, fidelity: float) -> pd.DataFrame:
    """Sobol indices over variable groups rather than single variables."""
    problem = {"num_vars": 5, "names": VARS, "bounds": [[0.0, 1.0]] * 5,
               "groups": groups}
    param = sobol_sample.sample(problem, N_SOBOL, calc_second_order=False,
                                seed=SEED)
    y = surrogate_mean(model, param, fidelity)
    Si = sobol_analyze.analyze(problem, y, calc_second_order=False,
                               print_to_console=False, seed=SEED)
    uniq = list(dict.fromkeys(groups))
    return pd.DataFrame({"group": uniq, "S1": Si["S1"], "S1_conf": Si["S1_conf"],
                         "ST": Si["ST"], "ST_conf": Si["ST_conf"]}
                        ).sort_values("ST", ascending=False)


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------
def plot_report(all_idx: pd.DataFrame, model, df: pd.DataFrame,
                support: pd.DataFrame, out_path: str) -> None:
    fig = plt.figure(figsize=(13.5, 9.5), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.95], hspace=0.42,
                          wspace=0.24)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    for ax in (ax_a, ax_b, ax_c):
        ax.set_facecolor(SURFACE)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#c3c2b7")
        ax.tick_params(colors=MUTED, labelsize=9)

    # ---- A: total-effect index per variable, one bar per fidelity ----------
    x = np.arange(len(VARS))
    w = 0.26
    for k, scale in enumerate(["micro", "bench", "pilot"]):
        sub = all_idx[all_idx["scale"] == scale].set_index("variable")
        vals = [sub.loc[v, "ST"] for v in VARS]
        ident = [bool(sub.loc[v, "identifiable"]) for v in VARS]
        pos = x + (k - 1) * w
        n = int(support[support.scale == scale]["n_runs"].iloc[0])
        ax_a.bar(pos, vals, w * 0.88, color=COLOR_OF[scale],
                 label=f"{scale} (n={n})", zorder=3)
        # Relief rule: direct-label every bar so identity/value never depends
        # on hue alone (slot 3 is sub-3:1 on the light surface).
        for xi, v, ok in zip(pos, vals, ident):
            ax_a.text(xi, v + 0.022, f"{v:.2f}", ha="center", va="bottom",
                      fontsize=7.5, color=INK2 if ok else MUTED,
                      fontweight="normal" if ok else "light")
            if not ok:
                # Hatch marks the index as unsupported by that scale's design.
                ax_a.bar(xi, v, w * 0.88, facecolor="none", edgecolor="white",
                         hatch="////", linewidth=0, zorder=4)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(VARS, color=INK, fontsize=11)
    ax_a.set_ylabel("total-effect index $S_T$", color=INK2, fontsize=10)
    ax_a.set_title(
        "Total effect on product amount, per reactor scale\n"
        "Hatched = the design at that scale never varied the input, so the "
        "index is extrapolated, not measured",
        color=INK, fontsize=11, loc="left")
    ax_a.grid(True, axis="y", color=GRID, lw=0.8, alpha=0.9, zorder=0)
    ax_a.set_axisbelow(True)
    ax_a.legend(frameon=False, fontsize=9, ncol=3, loc="upper left")
    ax_a.set_ylim(0, max(all_idx["ST"].max() * 1.22, 0.4))

    # Call out the artefact directly on the panel.
    pil = all_idx[(all_idx.scale == "pilot") & (all_idx.variable == "T")]
    mic = all_idx[(all_idx.scale == "micro") & (all_idx.variable == "T")]
    if len(pil) and len(mic):
        ax_a.annotate(
            f"T: $S_T$={float(mic.ST.iloc[0]):.2f} at micro (173 runs, full "
            f"20-60 C sweep)\nbut {float(pil.ST.iloc[0]):.3f} at pilot -- the 3 "
            f"pilot runs all sit at 34.6-40 C.\nThat 0.00 means 'never varied', "
            f"not 'does not matter'.",
            xy=(x[0] + w, 0.03), xytext=(x[0] + 0.55, 0.74),
            fontsize=8.5, color=INK2, ha="left",
            arrowprops=dict(arrowstyle="->", color=MUTED, lw=1,
                            connectionstyle="arc3,rad=0.15"))

    # ---- B: the GP's T profile at micro, with pilot coverage shaded --------
    grid = np.zeros((61, 5))
    grid[:, 0] = np.linspace(0, 1, 61)
    grid[:, 1] = (8.2 - 3) / 6.5      # pH at the optimum the loop converged to
    grid[:, 2] = 0.5
    grid[:, 3] = 0.5
    grid[:, 4] = 1.0                  # F3 saturated
    prof = surrogate_mean(model, grid, FIDELITY_OF["micro"])
    T_axis = 20 + 40 * grid[:, 0]
    ax_b.plot(T_axis, prof, color=COLOR_OF["micro"], lw=2, zorder=3)
    p = support[(support.scale == "pilot") & (support.variable == "T")].iloc[0]
    ax_b.axvspan(p["real_min"], p["real_max"], color="#c3c2b7", alpha=0.45,
                 zorder=1, lw=0)
    ax_b.annotate("all 3 pilot runs\nlive in here",
                  xy=((p["real_min"] + p["real_max"]) / 2, max(prof) * 0.30),
                  ha="center", fontsize=8.5, color=INK2)
    ax_b.set_xlabel("temperature T (deg C)", color=INK2, fontsize=10)
    ax_b.set_ylabel("surrogate mean $\\hat{Y}$ (micro)", color=INK2, fontsize=10)
    ax_b.set_title("The GP does know T matters -- at the scale with data\n"
                   "(pH 8.2, F3 2.0; peak at ~38 C)",
                   color=INK, fontsize=10.5, loc="left")
    ax_b.grid(True, color=GRID, lw=0.8, zorder=0)
    ax_b.set_axisbelow(True)

    # ---- C: raw micro data -- F3 only pays off inside the good T band ------
    m = df[df["scale"] == "micro"].copy()
    m["T_C"] = 20 + 40 * m["T_norm"]
    m["F3_r"] = 2 * m["F3_norm"]
    inside = (m["T_C"] > 35) & (m["T_C"] <= 40)
    edges = [0, 1.0, 1.5, 1.75, 2.0]
    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
    for lbl, mask, col in [("T in 35-40 C (optimal)", inside, "#2a78d6"),
                           ("T outside 35-40 C", ~inside, "#008300")]:
        s = m[mask]
        g = s.groupby(pd.cut(s["F3_r"], bins=edges), observed=False)["Y"].mean()
        ax_c.plot(centers, g.values, marker="o", ms=7, lw=2, color=col,
                  label=lbl, zorder=3)
        ax_c.annotate(f"{g.values[-1]:.2f}", xy=(centers[-1], g.values[-1]),
                      xytext=(5, 0), textcoords="offset points", va="center",
                      fontsize=8.5, color=INK2)
    ax_c.set_xlabel("F3 feed rate (g/L/h)", color=INK2, fontsize=10)
    ax_c.set_ylabel("mean observed Y (raw micro data)", color=INK2, fontsize=10)
    ax_c.set_title("Raw data, no model: F3 only pays off if T is right\n"
                   "=> T and F3 interact; the response is NOT additive",
                   color=INK, fontsize=10.5, loc="left")
    ax_c.grid(True, color=GRID, lw=0.8, zorder=0)
    ax_c.set_axisbelow(True)
    ax_c.legend(frameon=False, fontsize=9, loc="upper left")

    fig.savefig(out_path, dpi=140, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TAB_DIR, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)

    df = load_ref()
    model = fit_surrogate(df)
    support = design_support(df)

    # ---- indices at every fidelity ------------------------------------
    frames, second_frames = [], []
    for scale, fid in FIDELITY_OF.items():
        _, Si, n_pts = run_sobol_at(model, fid)
        frames.append(indices_frame(Si, scale, support))
        second_frames.append(second_order_frame(Si, scale))
    all_idx = pd.concat(frames, ignore_index=True)
    all_second = pd.concat(second_frames, ignore_index=True)
    print(f"evaluated surrogate at {n_pts} Saltelli points per fidelity")

    print("\n=== design support: did the campaign actually vary each input? ===")
    print("An index for an input with design_std < %.2f is extrapolated." %
          MIN_DESIGN_STD)
    piv = support.pivot(index="variable", columns="scale", values="design_std")
    print(piv.reindex(VARS)[["micro", "bench", "pilot"]].round(3).to_string())

    print("\n=== total-effect index ST, by fidelity ===")
    st = all_idx.pivot(index="variable", columns="scale", values="ST")
    print(st.reindex(VARS)[["micro", "bench", "pilot"]].round(3).to_string())

    # ---- the headline check ------------------------------------------
    raw = raw_effect_screen(df, "micro")
    pilot_idx = all_idx[all_idx.scale == "pilot"].set_index("variable")
    micro_idx = all_idx[all_idx.scale == "micro"].set_index("variable")
    pilot_sup = support[support.scale == "pilot"].set_index("variable")

    print("\n=== model-free screen of the RAW micro data (n=173, full sweep) ===")
    print("fold_change = max/min of mean Y across quantile bins of that input.")
    print("Compare fold_change against spearman_rho: rho misses a bump.")
    print(raw[["variable", "fold_change", "swing_abs",
               "spearman_rho"]].round(3).to_string(index=False))

    contradictions = []
    for v in VARS:
        r = raw[raw.variable == v]
        if not len(r):
            continue
        fold = float(r["fold_change"].iloc[0])
        st_p = float(pilot_idx.loc[v, "ST"])
        if st_p < INERT_ST and fold >= RAW_FOLD_ALERT:
            contradictions.append((v, st_p, fold,
                                   float(pilot_sup.loc[v, "real_min"]),
                                   float(pilot_sup.loc[v, "real_max"])))

    print("\n" + "=" * 72)
    if contradictions:
        print("*** THE PILOT-SLICE RANKING IS NOT A FINDING. ***")
        print("For these inputs the pilot GSA says 'inert' while the raw data "
              "says otherwise:\n")
        for v, st_p, fold, lo, hi in contradictions:
            lo_r, hi_r = REAL_RANGE[v]
            print(f"  {v:<3}: pilot ST = {st_p:.3f} (reads as inert), but raw "
                  f"micro data swings {fold:.1f}x across {v}.")
            print(f"       Cause: the 3 pilot runs only span {v} = {lo:.2f}-"
                  f"{hi:.2f} of the admissible {lo_r:.1f}-{hi_r:.1f}.")
            print(f"       The index is 0 because the design never moved {v}, "
                  f"not because {v} does nothing.\n")
        print("The BO loop converged onto the T/pH optimum, which removed the "
              "variance a\nvariance-based method needs. Report the micro-slice "
              "ranking (backed by 173\nfull-range runs) and treat the pilot "
              "slice as unidentifiable for these inputs.")
    else:
        print("No contradiction between the pilot indices and the raw-data "
              "screen.")
    print("=" * 72)

    # ---- what we CAN say ---------------------------------------------
    ident = micro_idx[micro_idx["identifiable"]].sort_values(
        "ST", ascending=False)
    print("\n=== defensible ranking: micro slice (every input fully swept) ===")
    print(ident[["ST", "S1", "design_std"]].round(3).to_string())
    print(f"\nRanking by total effect: {' > '.join(ident.index.tolist())}.")
    print("Read ST >> S1 as 'this variable acts mostly through interactions'.")

    micro_second = all_second[all_second.scale == "micro"]
    sig = micro_second[(micro_second["S2"] - micro_second["S2_conf"]) > 0]
    print("\n=== interactions at the micro slice ===")
    if len(sig):
        print("These pairs' bootstrap CIs exclude 0 -- tune them jointly, not "
              "one at a time:")
        print(sig[["pair", "S2", "S2_conf"]].round(3).to_string(index=False))
    else:
        print("Top pairs by S2 (none with CI excluding 0):")
        print(micro_second[["pair", "S2", "S2_conf"]].head(3).round(3)
              .to_string(index=False))
    tot_inter = float((micro_idx["ST"] - micro_idx["S1"]).sum())
    print(f"\nSum of (ST - S1) at micro = {tot_inter:.2f}. The earlier "
          f"'response is additive'\nclaim came from the pilot slice, where 3 "
          f"points cannot express an interaction.")

    # ---- collinearity, still worth reporting -------------------------
    coll = design_collinearity(df)
    print("\n=== caveat: Sobol assumes independent inputs ===")
    if len(coll):
        print("Correlated in the BO-selected design, so the S1/ST split between "
              "them is\nnot identifiable per-variable:")
        print(coll.round(3).to_string(index=False))
        groups = group_labels(coll)
        grouped = run_sobol_grouped(model, groups, TARGET_FIDELITY)
        print("\nGrouped indices at pilot (the split within a group is not "
              "identifiable, the group's total is):")
        print(grouped.round(3).to_string(index=False))
        grouped.to_csv(os.path.join(TAB_DIR, "task5_sobol_grouped.csv"),
                       index=False)
    else:
        print("No input pair exceeds |rho| = 0.4 in the design.")

    # ---- what to actually do -----------------------------------------
    print("\n=== recommended next experiments ===")
    if contradictions:
        names = ", ".join(v for v, *_ in contradictions)
        print(f"To make the pilot ranking identifiable, run pilot experiments "
              f"that vary\n{names} while holding the others at the converged "
              f"optimum:")
        for v, *_ in contradictions:
            lo_r, hi_r = REAL_RANGE[v]
            print(f"  - sweep {v} across {lo_r:g}-{hi_r:g} "
                  f"({REAL_RANGE[v][1] - REAL_RANGE[v][0]:g} wide), "
                  f"others fixed at the current best recipe")
        print("A handful of runs settles a several-fold effect; this is the one "
              "experiment\nthe campaign never performed.")

    # ---- persist ------------------------------------------------------
    all_idx.to_csv(os.path.join(TAB_DIR, "task5_sobol_first_total.csv"),
                   index=False)
    all_second.to_csv(os.path.join(TAB_DIR, "task5_sobol_second_order.csv"),
                      index=False)
    support.to_csv(os.path.join(TAB_DIR, "task5_design_support.csv"),
                   index=False)
    raw.to_csv(os.path.join(TAB_DIR, "task5_raw_effect_screen.csv"),
               index=False)
    fig_out = os.path.join(FIG_DIR, "task5_sensitivity.png")
    plot_report(all_idx, model, df, support, fig_out)
    print(f"\nsaved: {fig_out}")
    for t in ("task5_sobol_first_total.csv", "task5_sobol_second_order.csv",
              "task5_design_support.csv", "task5_raw_effect_screen.csv"):
        print(f"       {os.path.join(TAB_DIR, t)}")


if __name__ == "__main__":
    main()
