"""
Experiment 24: regenerate Fig. 2 (calibrated Stanghellini vs. CENTRAL sensor)
with the Jan-May 2019 calibration window shaded, so held-out and calibration
periods are visible. Same layout/scales as plot_results.py; the original
validation_plot.png is left untouched.
Output: revision_results/fig2_validation_calibration_shaded.png
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import rev_common as rc


def main(compact=False):
    p = rc.CONFIGS["Stanghellini (calibrated)"]
    # compact: shorter panels and no suptitle (the paper caption carries the title)
    fig, axes = plt.subplots(3, 3, figsize=(15, 5.6) if compact else (15, 8), sharex="col", sharey="row")
    if not compact:
        fig.suptitle("Hoogstraten Greenhouse: Simulated vs. Observed Indoor Climate (2018-2020)", fontsize=14)
    rows = [("T", "Indoor Temperature (°C)", (14, 36)),
            ("RH", "Indoor RH (%)", (20, 100)),
            ("VPD", "Indoor VPD (kPa)", (0, 4))]

    for j, year in enumerate(rc.YEARS):
        df = rc.run(year, p)
        for i, (v, label, ylim) in enumerate(rows):
            ax = axes[i, j]
            if year == 2019:
                ax.axvspan(df["time"].iloc[0], rc.CAL_END, color="0.85", zorder=0,
                           label="Calibration window" if i == 0 else None)
            ax.plot(df["time"], df[f"{v}_sim"], color="tab:blue", alpha=0.6, lw=0.5,
                    label="Simulated" if (i == 0 and j == 0) else None)
            ax.plot(df["time"], df[f"{v}_obs"], color="tab:red", alpha=0.8, lw=0.5,
                    label="Observed (CENTRAL sensor)" if (i == 0 and j == 0) else None)
            ax.set_ylim(*ylim)
            if j == 0:
                ax.set_ylabel(label)
            if i == 0:
                ax.set_title(str(year))
                if j in (0, 1):
                    ax.legend(loc="upper right", fontsize="small")
        axes[2, j].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        axes[2, j].xaxis.set_major_locator(mdates.MonthLocator())

    plt.tight_layout(rect=[0, 0, 1, 1 if compact else 0.96])
    name = "fig2_validation_compact.png" if compact else "fig2_validation_calibration_shaded.png"
    out = os.path.join(rc.OUT_DIR, name)
    plt.savefig(out, dpi=300)
    print("Saved", out)


if __name__ == "__main__":
    import sys
    main(compact="--compact" in sys.argv)
