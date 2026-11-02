"""
Experiment 15 (Reviewer 1: "additional validation metrics, distributional
comparison, seasonal/hourly error analysis"; also backs Reviewer 2 point 6 on
night-time errors).

All on the held-out window (2018, Jun-Sep 2019, 2020), pooled hourly data.
Day = outdoor GHI > 0 W/m2, night = GHI == 0.
Outputs: revision_results/exp15_*.csv and exp15_hourly_bias.png
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance, ks_2samp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import rev_common as rc

CONFIG_NAMES = ["Stanghellini (calibrated)", "Stanghellini (literature)",
                "Priestley-Taylor (zero-data)", "FAO-56 PM (zero-data)"]
VARS = [("T", "°C"), ("RH", "%"), ("VPD", "kPa")]


def heldout_pooled(p):
    return pd.concat([rc.window(rc.run(y, p), y, "heldout") for y in rc.YEARS], ignore_index=True)


def main():
    data = {n: heldout_pooled(rc.CONFIGS[n]) for n in CONFIG_NAMES}
    pd.set_option("display.width", 220)

    # 1. Summary: RMSE, MAE, bias, EF (mean of per-year metrics, as in the paper)
    rows = []
    for n in CONFIG_NAMES:
        mean = rc.heldout_scores(rc.CONFIGS[n])[1]
        rows.append(dict(config=n, **{k: mean[k] for k in mean.index if not k.endswith("_n")}))
    summary = pd.DataFrame(rows).set_index("config")
    summary.to_csv(os.path.join(rc.OUT_DIR, "exp15_summary_metrics.csv"))
    print("1) Held-out RMSE / MAE / bias / EF (mean of per-year):")
    print(summary.round(2).to_string())

    # 2. Day vs night
    rows = []
    for n, d in data.items():
        for label, mask in [("day", d["shortwave_radiation"] > 0), ("night", d["shortwave_radiation"] <= 0)]:
            s = rc.score(d[mask])
            rows.append(dict(config=n, period=label, hours=int(mask.sum()),
                             **{f"{v}_{m}": s[f"{v}_{m}"] for v, _ in VARS for m in ["rmse", "bias"]}))
    daynight = pd.DataFrame(rows)
    daynight.to_csv(os.path.join(rc.OUT_DIR, "exp15_day_night.csv"), index=False)
    print("\n2) Day vs night (pooled held-out):")
    print(daynight.round(2).to_string(index=False))

    # 3. Hour-of-day bias
    hourly = []
    for n, d in data.items():
        g = d.assign(hour=d["time"].dt.hour)
        for v, _ in VARS:
            b = (g[f"{v}_sim"] - g[f"{v}_obs"]).groupby(g["hour"]).mean()
            hourly.append(pd.DataFrame({"config": n, "var": v, "hour": b.index, "bias": b.values}))
    hourly = pd.concat(hourly)
    hourly.to_csv(os.path.join(rc.OUT_DIR, "exp15_hourly_bias.csv"), index=False)
    print("\n3) Hour-of-day bias range (min..max over 24 h):")
    print(hourly.groupby(["config", "var"])["bias"].agg(["min", "max"]).round(2).to_string())

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
    for ax, (v, unit) in zip(axes, VARS):
        for n in CONFIG_NAMES:
            h = hourly[(hourly.config == n) & (hourly["var"] == v)]
            ax.plot(h["hour"], h["bias"], marker="o", ms=3, label=n)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"{v} bias (sim − obs, {unit})")
        ax.set_xlabel("Hour of day")
        ax.set_xticks(range(0, 24, 3))
    axes[0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(rc.OUT_DIR, "exp15_hourly_bias.png"), dpi=200)

    # 4. Monthly RMSE
    rows = []
    for n, d in data.items():
        for (y, m), g in d.groupby([d["time"].dt.year, d["time"].dt.month]):
            s = rc.score(g)
            rows.append(dict(config=n, year=y, month=m, hours=len(g),
                             T_rmse=s["T_rmse"], RH_rmse=s["RH_rmse"], VPD_rmse=s["VPD_rmse"]))
    monthly = pd.DataFrame(rows)
    monthly.to_csv(os.path.join(rc.OUT_DIR, "exp15_monthly.csv"), index=False)
    print("\n4) Monthly RMSE, calibrated Stanghellini (averaged over years per calendar month):")
    print(monthly[monthly.config == CONFIG_NAMES[0]].groupby("month")[["T_rmse", "RH_rmse", "VPD_rmse"]]
          .mean().round(2).to_string())

    # 5. Distributions: quantiles, Wasserstein distance, KS statistic
    rows = []
    for n, d in data.items():
        for v, _ in VARS:
            o = d[f"{v}_obs"].dropna().to_numpy()
            s = d[f"{v}_sim"].dropna().to_numpy()
            q = [5, 25, 50, 75, 95]
            row = dict(config=n, var=v, wasserstein=wasserstein_distance(o, s),
                       ks_stat=ks_2samp(o, s).statistic)
            for qq, a, b in zip(q, np.percentile(o, q), np.percentile(s, q)):
                row[f"obs_p{qq}"] = a
                row[f"sim_p{qq}"] = b
            rows.append(row)
    dist = pd.DataFrame(rows)
    dist.to_csv(os.path.join(rc.OUT_DIR, "exp15_distributions.csv"), index=False)
    print("\n5) Distribution comparison (pooled held-out):")
    print(dist[["config", "var", "wasserstein", "ks_stat", "obs_p5", "sim_p5", "obs_p50",
                "sim_p50", "obs_p95", "sim_p95"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
