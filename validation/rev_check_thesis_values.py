"""
Check: literature Stanghellini with the paper's rounded values (ra 250,
rs_min 100, rs_dark 400) vs. the values in Stanghellini (1987) for greenhouse
tomato: external resistance ~200 s/m (pp. 80, 85-86), r_min = 82 s/m and
dark multiplier C1/C2 = 4.30/0.54 (Table 3.3, p. 90) -> ~653 s/m in the dark.
Held-out window, mean of per-year metrics.
"""
import pandas as pd

import rev_common as rc

CONFIGS = {
    "paper values (250/100/400)": rc.CONFIGS["Stanghellini (literature)"],
    "thesis values (200/82/653)": dict(mode=1, ach_open=rc.LIT_ACH, ra=200.0, rs_min=82.0,
                                       rs_dark=82.0 * 4.30 / 0.54),
}

if __name__ == "__main__":
    rows = {k: rc.heldout_scores(p)[1][["T_rmse", "T_ef", "RH_rmse", "RH_ef", "VPD_rmse", "VPD_ef", "RH_bias"]]
            for k, p in CONFIGS.items()}
    print(pd.DataFrame(rows).T.round(2).to_string())
