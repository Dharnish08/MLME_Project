"""Additional task 4: visualise the per-iteration scale-selection pattern.

The multi-fidelity BO loop in ``mfbo_v1.py`` chooses, at every iteration, BOTH
a recipe and a *scale* (micro / bench / pilot).  Two things drive that choice:

  1. The cost-aware Knowledge-Gradient acquisition (``qMultiFidelityKnowledge
     Gradient`` weighted by ``InverseCostWeightedUtility``).  Because micro runs
     are ~200x cheaper than pilot runs, the acquisition almost always prefers
     the cheap micro scale unless a more expensive run buys a lot of
     information about the pilot optimum.

  2. Two hard-coded *interventions* that keep the loop honest:
        * after a streak of 20 micro runs  -> force one bench calibration
        * after 80 iterations without pilot -> force one pilot run at the
          current predicted optimum.

This script does NOT call the API.  It replays a finished campaign log and
reconstructs, iteration by iteration, whether each scale choice was made
*voluntarily* by the acquisition or was *forced* by an intervention, using the
exact counter logic from ``mfbo_v1.py``.  It then plots the scale timeline
together with cumulative cost and the running best pilot yield, so the drivers
of every scale transition are visible.

Takes a *cleaned* log (with `scale`, `Y`, `cumulative_cost_eur`), not a raw
`results/optimization_log_*.csv` -- those use the older writer's schema and are
rejected with an explanatory error.

    python task4_scale_selection.py [path/to/cleaned_log.csv]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOG = os.path.join(
    HERE, "results", "tables_initial_experiments",
    "cleaned_full_optimization_log_reference.csv")
FIG_DIR = os.path.join(HERE, "results", "figures_additional_tasks")
TAB_DIR = os.path.join(HERE, "results", "tables_additional_tasks")

# Same thresholds as mfbo_v1.py.
MICRO_STREAK_LIMIT = 20     # force a bench run after this many micros in a row
NO_PILOT_LIMIT = 80         # force a pilot run after this many steps w/o pilot

SCALE_OF_IDX = {0: "micro", 1: "bench", 2: "pilot"}
IDX_OF_SCALE = {v: k for k, v in SCALE_OF_IDX.items()}
COLOR = {"micro": "#4C9BE8", "bench": "#F0A030", "pilot": "#D64550"}


def load_log(path: str) -> pd.DataFrame:
    """Load a *cleaned* campaign log.

    The raw ``results/optimization_log_*.csv`` files use the older writer's
    schema (``Targetvalue_Y`` / ``Cost_Cummulated``, no ``scale``), so they are
    not interchangeable with the cleaned reference this script expects. Say so
    up front rather than dying on a bare KeyError deep in the plot.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]          # some logs are padded
    if "Fidelity_Index" not in df.columns:
        df["Fidelity_Index"] = (df["Fidelity_Scaled"] * 2).round().astype(int)
    if "scale" not in df.columns:
        df["scale"] = df["Fidelity_Index"].map(SCALE_OF_IDX)
    df["Fidelity_Index"] = df["Fidelity_Index"].astype(int)

    missing = [c for c in ("Y", "cumulative_cost_eur") if c not in df.columns]
    if missing:
        raise SystemExit(
            f"{os.path.basename(path)} is missing {missing}.\n"
            f"It looks like a RAW optimization log (its schema uses "
            f"'Targetvalue_Y'/'Cost_Cummulated').\nThis script expects a "
            f"cleaned log with Y and cumulative_cost_eur -- e.g.\n"
            f"  {os.path.relpath(DEFAULT_LOG, HERE)}\n"
            f"Raw logs must be cleaned first (see visualization_and_bias.py).")
    return df


def reconstruct_decisions(bo: pd.DataFrame) -> pd.DataFrame:
    """Replay mfbo_v1's counter logic to label each BO step forced/voluntary.

    Mirrors exactly the branch structure of the main loop: the counters are
    read *before* a step to decide whether an intervention fires, then updated
    from the fidelity that was actually recorded.
    """
    micro_streak = 0
    steps_without_pilot = 0
    labels = []
    for fid in bo["Fidelity_Index"].to_numpy():
        # ---- which branch fired (state BEFORE the step) ----
        if steps_without_pilot >= NO_PILOT_LIMIT:
            decision = "forced-pilot"
        elif micro_streak >= MICRO_STREAK_LIMIT:
            decision = "forced-bench"
        else:
            decision = "voluntary"
        labels.append(decision)

        # ---- update counters exactly as mfbo_v1 does ----
        if decision == "forced-pilot":
            steps_without_pilot = 0
            micro_streak = 0
        elif decision == "forced-bench":
            micro_streak = 0
            steps_without_pilot = min(steps_without_pilot + 1, 70)
        else:  # voluntary: react to what the acquisition actually picked
            if fid == 2:        # pilot
                steps_without_pilot = 0
                micro_streak = 0
            elif fid == 1:      # bench
                steps_without_pilot += 1
                micro_streak = 0
            else:               # micro
                steps_without_pilot += 1
                micro_streak += 1

    out = bo.copy()
    out["decision"] = labels

    # The counters are replayed blind: they assert "an intervention fired here"
    # from step count alone, without consulting what the log actually records.
    # If a log was produced by a build whose intervention logic differs (or has
    # none), that assertion invents forced runs that never happened. Check the
    # claim against the recorded fidelity instead of trusting it.
    expected = {"forced-bench": 1, "forced-pilot": 2}
    mismatch = out.apply(
        lambda r: r["decision"] in expected
        and r["Fidelity_Index"] != expected[r["decision"]], axis=1)
    out.attrs["n_mismatch"] = int(mismatch.sum())
    if mismatch.any():
        out.loc[mismatch, "decision"] = "unverified"
    return out


def transition_table(bo: pd.DataFrame) -> pd.DataFrame:
    """Count scale->scale transitions across consecutive BO steps."""
    scales = bo["scale"].tolist()
    rows = []
    for a, b in zip(scales[:-1], scales[1:]):
        if a != b:
            rows.append((a, b))
    if not rows:
        return pd.DataFrame(columns=["from", "to", "count"])
    tt = (pd.DataFrame(rows, columns=["from", "to"])
          .value_counts().rename("count").reset_index())
    return tt.sort_values("count", ascending=False)


def plot(bo: pd.DataFrame, out_path: str, init_best_pilot: float) -> None:
    """Three stacked single-axis panels sharing the iteration axis.

    Each panel carries exactly one quantity: no twin y-axes (a dual-scale plot
    invites reading a crossing point that means nothing), no per-point
    annotations, no connecting line between discrete scale choices.
    """
    it = bo["Iteration"].to_numpy()
    fid = bo["Fidelity_Index"].to_numpy()
    dec = bo["decision"].to_numpy()
    # Only ring points whose forced label was confirmed against the log; an
    # "unverified" step must not be drawn as a confirmed intervention.
    forced = np.isin(dec, ["forced-bench", "forced-pilot"])

    fig, (ax_s, ax_c, ax_y) = plt.subplots(
        3, 1, figsize=(11, 8.5), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 0.8, 0.9]})

    # ---- panel 1: which scale was chosen -------------------------------
    ax_s.scatter(it[~forced], fid[~forced],
                 c=[COLOR[s] for s in bo["scale"][~forced]],
                 s=16, linewidths=0, zorder=3)
    ax_s.scatter(it[forced], fid[forced],
                 c=[COLOR[s] for s in bo["scale"][forced]],
                 s=70, edgecolors="black", linewidths=1.0, zorder=4)
    ax_s.set_yticks([0, 1, 2])
    ax_s.set_yticklabels(["micro\n(10 EUR)", "bench\n(500 EUR)",
                          "pilot\n(2000 EUR)"])
    ax_s.set_ylim(-0.5, 2.5)
    title = "Scale chosen at each BO iteration"
    if bo.attrs.get("n_mismatch", 0):
        title += ("\n(forced/voluntary split UNRELIABLE: this log does not "
                  "match mfbo_v1's intervention logic)")
    ax_s.set_title(title, fontsize=10)
    ax_s.grid(True, axis="x", alpha=0.2)

    # colour = scale, outline = forced; spelled out so identity is never
    # carried by colour alone.
    handles = [Line2D([], [], marker="o", ls="", ms=5, mfc=COLOR[s],
                      mec="none", label=s) for s in ["micro", "bench", "pilot"]]
    handles.append(Line2D([], [], marker="o", ls="", ms=8, mfc="0.85",
                          mec="black", mew=1.0,
                          label="forced by intervention"))
    ax_s.legend(handles=handles, loc="upper left", fontsize=8,
                ncol=4, framealpha=0.9)

    # ---- panel 2: what it cost -----------------------------------------
    ax_c.plot(it, bo["cumulative_cost_eur"], color="#6A3D9A", lw=2)
    ax_c.set_ylabel("cumulative cost\n(EUR)")
    ax_c.grid(True, alpha=0.2)

    # ---- panel 3: what it bought ---------------------------------------
    # Running best pilot yield must carry the initial design's pilots forward,
    # otherwise it reads as "no pilot data" until the first BO pilot fires.
    pilot_y = bo["Y"].where(bo["Fidelity_Index"] == 2)
    running_best = pilot_y.cummax().ffill()
    if np.isfinite(init_best_pilot):
        running_best = running_best.fillna(init_best_pilot).clip(
            lower=init_best_pilot)
    ax_y.step(it, running_best, where="post", color=COLOR["pilot"], lw=2,
              label="best pilot Y so far")
    if "Pred_Max_Y" in bo.columns:
        ax_y.plot(it, pd.to_numeric(bo["Pred_Max_Y"], errors="coerce"),
                  color="0.35", lw=1.3, ls="--",
                  label="surrogate predicted max Y")
    ax_y.set_ylabel("pilot yield Y")
    ax_y.set_xlabel("BO iteration")
    ax_y.grid(True, alpha=0.2)
    ax_y.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    log_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LOG
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TAB_DIR, exist_ok=True)

    df = load_log(log_path)
    init = df[df["Iteration"] == 0]
    bo = df[df["Iteration"] > 0].reset_index(drop=True)
    bo = reconstruct_decisions(bo)

    # The initial design already contains pilot runs; the BO loop's "best pilot
    # so far" starts from them, not from nothing.
    init_pilot = init.loc[init["scale"] == "pilot", "Y"]
    init_best_pilot = float(init_pilot.max()) if len(init_pilot) else np.nan

    # ---- console summary ----
    print(f"log: {os.path.basename(log_path)}")
    print(f"initial design: {len(init)} runs "
          f"({init['scale'].value_counts().to_dict()})")
    if len(init_pilot):
        print(f"  best pilot Y in initial design: {init_best_pilot:.3f} "
              f"({len(init_pilot)} pilot runs) -- carried into the BO curve")
    print(f"BO iterations : {len(bo)}")
    print("\nscale usage during BO:")
    print(bo["scale"].value_counts().reindex(
        ["micro", "bench", "pilot"]).fillna(0).astype(int).to_string())
    print("\ndecision breakdown:")
    print(bo["decision"].value_counts().to_string())

    n_mismatch = bo.attrs.get("n_mismatch", 0)
    if n_mismatch:
        print(f"\n*** WARNING: this log does NOT match mfbo_v1's intervention "
              f"logic. ***\nReplaying the counters claims an intervention fired "
              f"at {n_mismatch} step(s) where the log\nactually records a "
              f"different scale -- i.e. those interventions never happened.\n"
              f"This log was probably produced by a build with different (or "
              f"no) interventions,\nso the forced/voluntary split is NOT "
              f"reliable here; those steps are marked\n'unverified' rather than "
              f"invented. Scale usage and cost above are still valid.")
    else:
        n_forced = (bo["decision"] != "voluntary").sum()
        print(f"\n{n_forced}/{len(bo)} BO scale choices were forced by an "
              f"intervention. (Every forced label was\nchecked against the "
              f"scale the log actually records.)")

        # What the acquisition does when left alone is the whole question here,
        # so measure it rather than assert it: if every voluntary step is micro,
        # then the interventions -- not the acquisition -- author every scale
        # transition in the figure above.
        vol = bo[bo["decision"] == "voluntary"]
        vol_mix = vol["scale"].value_counts()
        n_vol_micro = int(vol_mix.get("micro", 0))
        if len(vol) and n_vol_micro == len(vol):
            print(f"\nEvery one of the {len(vol)} voluntary choices was micro: "
                  f"the cost-aware acquisition\nNEVER paid for bench or pilot "
                  f"on its own. Inverse-cost weighting makes a pilot run\nneed "
                  f"~200x the KG of a micro run to win, and it never does -- so "
                  f"every scale\ntransition in the figure was authored by an "
                  f"intervention, not by the acquisition.\nThe interventions are "
                  f"not a safety net here; they are the only reason this loop\n"
                  f"ever sees the pilot scale it is optimising for.")
        elif len(vol):
            print(f"\nVoluntary choices by scale: {vol_mix.to_dict()} -- the "
                  f"acquisition paid for a\nnon-micro run {len(vol) - n_vol_micro}"
                  f" time(s) without being forced.")

    tt = transition_table(bo)
    print("\nscale transitions (consecutive BO steps):")
    print(tt.to_string(index=False) if len(tt) else "  (none)")

    # ---- persist ----
    bo_out = os.path.join(TAB_DIR, "task4_scale_decisions.csv")
    bo[["Iteration", "scale", "Fidelity_Index", "decision", "Y",
        "cumulative_cost_eur"]].to_csv(bo_out, index=False)
    tt_out = os.path.join(TAB_DIR, "task4_scale_transitions.csv")
    tt.to_csv(tt_out, index=False)
    fig_out = os.path.join(FIG_DIR, "task4_scale_selection.png")
    plot(bo, fig_out, init_best_pilot)
    print(f"\nsaved: {fig_out}\n       {bo_out}\n       {tt_out}")


if __name__ == "__main__":
    main()
