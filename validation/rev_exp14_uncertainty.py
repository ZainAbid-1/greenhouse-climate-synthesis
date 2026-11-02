"""
Experiment 14 (Reviewer 1: "no uncertainty quantification beyond comparing
formulations").

A) Weekly block bootstrap (1000 resamples) of the pooled held-out hours.
   The same resampled weeks are used for every configuration (paired), so the
   CI on the DIFFERENCE between two configurations shows whether a ranking is
   robust. Point estimates here are pooled over all held-out hours, so they
   differ slightly from the paper's mean-of-per-year values.
B) One-at-a-time +-20% perturbation of the four calibrated Stanghellini
   parameters, scored on the held-out window (mean of per-year).
Outputs: revision_results/exp14_*.csv
"""
import os
import numpy as np
import pandas as pd

import rev_common as rc

CONFIG_NAMES = ["Stanghellini (calibrated)", "Stanghellini (literature)",
                "Priestley-Taylor (zero-data)", "FAO-56 PM (zero-data)"]
VARS = ["T", "RH", "VPD"]
N_BOOT = 1000
SEED = 418


def weekly_sums(d, v):
    """Per-week sufficient statistics so each bootstrap draw is a cheap sum."""
    o = d[f"{v}_obs"].to_numpy()
    s = d[f"{v}_sim"].to_numpy()
    ok = ~(np.isnan(o) | np.isnan(s))
    w = d["week"].to_numpy()[ok]
    o, s = o[ok], s[ok]
    g = pd.DataFrame({"w": w, "n": 1.0, "sse": (s - o) ** 2, "so": o, "so2": o ** 2}).groupby("w").sum()
    return g


def stats_from(g):
    n, sse, so, so2 = (g[c].to_numpy() for c in ["n", "sse", "so", "so2"])
    return n, sse, so, so2


def main():
    rng = np.random.default_rng(SEED)
    data = {}
    for n in CONFIG_NAMES:
        d = pd.concat([rc.window(rc.run(y, rc.CONFIGS[n]), y, "heldout") for y in rc.YEARS],
                      ignore_index=True)
        d["week"] = d["time"].dt.strftime("%G-%V")
        data[n] = d
    weeks = sorted(data[CONFIG_NAMES[0]]["week"].unique())
    draws = rng.integers(0, len(weeks), size=(N_BOOT, len(weeks)))
    print(f"A) Block bootstrap: {len(weeks)} held-out weeks, {N_BOOT} resamples (paired across configs)")

    boot = {}   # (config, var) -> (rmse array, ef array, point rmse, point ef)
    for n, d in data.items():
        for v in VARS:
            g = weekly_sums(d, v).reindex(weeks).fillna(0.0)
            cnt, sse, so, so2 = stats_from(g)
            N = cnt[draws].sum(1); SSE = sse[draws].sum(1)
            SO = so[draws].sum(1); SO2 = so2[draws].sum(1)
            rmse = np.sqrt(SSE / N)
            ef = 1.0 - SSE / (SO2 - SO ** 2 / N)
            p_rmse = np.sqrt(sse.sum() / cnt.sum())
            p_ef = 1.0 - sse.sum() / (so2.sum() - so.sum() ** 2 / cnt.sum())
            boot[(n, v)] = (rmse, ef, p_rmse, p_ef)

    rows = []
    for (n, v), (rmse, ef, pr, pe) in boot.items():
        rows.append(dict(config=n, var=v, rmse=pr, rmse_lo=np.percentile(rmse, 2.5),
                         rmse_hi=np.percentile(rmse, 97.5), ef=pe,
                         ef_lo=np.percentile(ef, 2.5), ef_hi=np.percentile(ef, 97.5)))
    ci = pd.DataFrame(rows)
    ci.to_csv(os.path.join(rc.OUT_DIR, "exp14_bootstrap_ci.csv"), index=False)
    pd.set_option("display.width", 220)
    print(ci.round(2).to_string(index=False))

    pairs = [("Stanghellini (calibrated)", "Stanghellini (literature)"),
             ("Stanghellini (literature)", "Priestley-Taylor (zero-data)"),
             ("Stanghellini (literature)", "FAO-56 PM (zero-data)"),
             ("FAO-56 PM (zero-data)", "Priestley-Taylor (zero-data)")]
    rows = []
    for a, b in pairs:
        for v in VARS:
            diff = boot[(a, v)][0] - boot[(b, v)][0]
            rows.append(dict(a=a, b=b, var=v, rmse_diff=boot[(a, v)][2] - boot[(b, v)][2],
                             lo=np.percentile(diff, 2.5), hi=np.percentile(diff, 97.5),
                             share_a_better=float(np.mean(diff < 0))))
    diffs = pd.DataFrame(rows)
    diffs.to_csv(os.path.join(rc.OUT_DIR, "exp14_bootstrap_differences.csv"), index=False)
    print("\nRMSE difference a - b (negative = a better), 95% CI:")
    print(diffs.round(3).to_string(index=False))

    # B) parameter perturbation
    base = rc.CONFIGS["Stanghellini (calibrated)"]
    rows = []
    _, m0 = rc.heldout_scores(base)
    rows.append(dict(param="(none)", change="0%", **m0[["T_rmse", "T_ef", "RH_rmse", "RH_ef", "VPD_rmse", "VPD_ef"]]))
    for k in ["ach_open", "ra", "rs_min", "rs_dark"]:
        for f in [0.8, 1.2]:
            p = dict(base)
            p[k] = base[k] * f
            _, m = rc.heldout_scores(p)
            rows.append(dict(param=k, change=f"{(f - 1) * 100:+.0f}%",
                             **m[["T_rmse", "T_ef", "RH_rmse", "RH_ef", "VPD_rmse", "VPD_ef"]]))
    pert = pd.DataFrame(rows)
    pert.to_csv(os.path.join(rc.OUT_DIR, "exp14_parameter_perturbation.csv"), index=False)
    print("\nB) +-20% perturbation of calibrated Stanghellini parameters (held-out, mean of per-year):")
    print(pert.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
