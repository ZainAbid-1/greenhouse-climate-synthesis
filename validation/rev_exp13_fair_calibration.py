"""
Experiment 13 (Reviewer 2, point 2): calibrate EACH transpiration formulation
on the same window with the same objective, then score all on the held-out
window -- separating the effect of formulation from the effect of fitting.

Calibration window : 2019-01-02 .. 2019-05-31 (as in the paper)
Objective          : maximise mean(T EF, RH EF) on that window (same for all)
Search             : exhaustive grid search
Outputs            : revision_results/exp13_*.csv
"""
import itertools
import os
import time
import pandas as pd

import rev_common as rc

GRIDS = {
    # name: (mode, {param: values})
    "Priestley-Taylor (fitted)": (0, dict(
        alpha_pt=[0.5, 0.75, 1.0, 1.26, 1.5, 2.0, 2.5, 3.0, 4.0],
        trans_floor=[0.0, 25000.0, 50000.0, 75000.0, 100000.0],
        ach_open=[5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 90.0, 120.0])),
    "FAO-56 PM (fitted)": (2, dict(
        rs_fao=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0, 70.0, 100.0, 150.0, 200.0, 300.0],
        ach_open=[5.0, 10.0, 15.0, 20.0, 30.0, 45.0])),
    "Stanghellini (fitted)": (1, dict(
        ra=[5.0, 10.0, 20.0, 40.0, 80.0, 150.0, 250.0],
        rs_min=[5.0, 10.0, 25.0, 50.0, 100.0],
        rs_dark=[100.0, 200.0, 350.0, 500.0, 800.0],
        ach_open=[5.0, 10.0, 15.0, 20.0, 30.0, 45.0])),
}


def cal_objective(p):
    d = rc.window(rc.run(2019, p), 2019, "calibration")
    s = rc.score(d)
    return 0.5 * (s["T_ef"] + s["RH_ef"]), s


def main():
    all_rows, best = [], {}
    for name, (mode, grid) in GRIDS.items():
        keys = list(grid)
        t0 = time.time()
        rows = []
        for vals in itertools.product(*grid.values()):
            p = dict(mode=mode, **dict(zip(keys, vals)))
            obj, s = cal_objective(p)
            rows.append(dict(config=name, objective=obj, **dict(zip(keys, vals)),
                             cal_T_rmse=s["T_rmse"], cal_T_ef=s["T_ef"],
                             cal_RH_rmse=s["RH_rmse"], cal_RH_ef=s["RH_ef"]))
        df = pd.DataFrame(rows).sort_values("objective", ascending=False)
        all_rows.append(df)
        top = df.iloc[0]
        p = dict(mode=mode, **{k: float(top[k]) for k in keys})
        on_edge = [k for k in keys if top[k] in (min(grid[k]), max(grid[k]))]
        best[name] = (p, len(keys), on_edge)
        print(f"{name}: {len(df)} runs in {time.time() - t0:.0f}s; best "
              + ", ".join(f"{k}={top[k]:g}" for k in keys)
              + f"  (cal objective {top['objective']:.3f})"
              + (f"  [on grid edge: {', '.join(on_edge)}]" if on_edge else ""))

    pd.concat(all_rows).to_csv(os.path.join(rc.OUT_DIR, "exp13_grid_search.csv"), index=False)

    # paper's calibrated Stanghellini, scored with the same objective for reference
    paper = rc.CONFIGS["Stanghellini (calibrated)"]
    obj, _ = cal_objective(paper)
    print(f"Paper's calibrated Stanghellini set: cal objective {obj:.3f}")

    # held-out comparison: zero-data vs fitted, same window for all
    table = {}
    order = [
        ("Priestley-Taylor (zero-data)", rc.CONFIGS["Priestley-Taylor (zero-data)"], 0),
        ("FAO-56 PM (zero-data)", rc.CONFIGS["FAO-56 PM (zero-data)"], 0),
        ("Stanghellini (literature)", rc.CONFIGS["Stanghellini (literature)"], 0),
        ("Priestley-Taylor (fitted)",) + best["Priestley-Taylor (fitted)"][:2],
        ("FAO-56 PM (fitted)",) + best["FAO-56 PM (fitted)"][:2],
        ("Stanghellini (fitted)",) + best["Stanghellini (fitted)"][:2],
        ("Stanghellini (paper calibration)", paper, 4),
    ]
    per_year_all = []
    for name, p, nfit in order:
        per_year, mean = rc.heldout_scores(p)
        per_year.insert(0, "config", name)
        per_year_all.append(per_year)
        row = mean[["T_rmse", "T_ef", "RH_rmse", "RH_ef", "VPD_rmse", "VPD_ef"]].to_dict()
        row["fitted_params"] = nfit
        row["params"] = ", ".join(f"{k}={v:g}" for k, v in p.items() if k != "mode")
        table[name] = row
    tab = pd.DataFrame(table).T
    tab.to_csv(os.path.join(rc.OUT_DIR, "exp13_heldout_comparison.csv"))
    pd.concat(per_year_all).to_csv(os.path.join(rc.OUT_DIR, "exp13_heldout_per_year.csv"), index=False)
    pd.set_option("display.width", 220)
    print("\nHeld-out (2018, Jun-Sep 2019, 2020), mean of per-year metrics:")
    print(tab[["fitted_params", "T_rmse", "T_ef", "RH_rmse", "RH_ef", "VPD_rmse", "VPD_ef"]]
          .apply(pd.to_numeric).round(2).to_string())
    print("\nParameters:")
    for k, r in table.items():
        print(f"  {k:<34} {r['params']}")


if __name__ == "__main__":
    main()
