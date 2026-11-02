"""
Shared helpers for the camera-ready revision experiments (rev_*.py).

Does NOT modify model.py / driver.py. It reuses their constants and data
loaders, adds forcing caching (so grid searches don't re-read the Excel files
on every run), and provides `simulate_ext`: a copy of model.simulate with two
extra arguments (alpha_pt, f_net_rad) that are hard-coded in model.py.
`check_reproduction()` verifies simulate_ext matches model.simulate exactly
at the original values (alpha_pt=1.26, f_net_rad=model.F_NET_RAD).

Evaluation windows
  calibration : 2019-01-02 .. 2019-05-31
  held-out    : 2018 (all), 2019-06-01 .. 2019-10-01, 2020 (all)
"""
import os
import numpy as np
import pandas as pd
from numba import njit

import model
import driver

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "revision_results")
os.makedirs(OUT_DIR, exist_ok=True)

YEARS = [2018, 2019, 2020]
CAL_END = pd.Timestamp("2019-06-01")
SUB_STEPS = 120

# ------------------------------------------------------------------------------
# Configurations
#   "paper" = exactly as used for the submitted Table III
#   "zero"  = genuinely zero-indoor-data (literature ACH 30, no fitted floor)
# ------------------------------------------------------------------------------
LIT_ACH = 30.0
CONFIGS = {
    "Stanghellini (calibrated)":       dict(mode=1, ach_open=15.0, ra=20.0, rs_min=10.0, rs_dark=350.0),
    "Stanghellini (literature)":       dict(mode=1, ach_open=LIT_ACH, ra=250.0, rs_min=100.0, rs_dark=400.0),
    "Priestley-Taylor (zero-data)":    dict(mode=0, ach_open=LIT_ACH),
    "FAO-56 PM (zero-data)":           dict(mode=2, ach_open=LIT_ACH),
    "Priestley-Taylor (as submitted)": dict(mode=0, ach_open=15.0, trans_floor=50000.0),
    "FAO-56 PM (as submitted)":        dict(mode=2, ach_open=15.0),
}


def full_params(p):
    """Fill every simulate argument; unspecified ones take model.py defaults."""
    return dict(
        mode=p["mode"],
        ach_closed=p.get("ach_closed", model.ACH_CLOSED),
        ach_open=p.get("ach_open", model.ACH_OPEN),
        ra=p.get("ra", model.RA),
        rs_min=p.get("rs_min", model.RS_MIN),
        rs_dark=p.get("rs_dark", model.RS_DARK),
        rs_r0=p.get("rs_r0", model.RS_R0),
        rs_fao=p.get("rs_fao", model.RS_FAO),
        u_min_fao=p.get("u_min_fao", model.U_MIN_FAO),
        u_max_fao=p.get("u_max_fao", model.U_MAX_FAO),
        trans_floor=p.get("trans_floor", 0.0),
        alpha_pt=p.get("alpha_pt", 1.26),
        f_net_rad=p.get("f_net_rad", model.F_NET_RAD),
    )


# ------------------------------------------------------------------------------
# simulate_ext: model.simulate + alpha_pt, f_net_rad as arguments
# (body identical to model.simulate otherwise)
# ------------------------------------------------------------------------------
@njit(cache=True)
def simulate_ext(n, sub_steps, Tout, G, P_kpa, hour_local, rho_v_out,
                 transpiration_mode, ach_closed, ach_open, ra, rs_min, rs_dark, rs_r0,
                 rs_fao, u_min_fao, u_max_fao, trans_floor, alpha_pt, f_net_rad):
    AF = 1600.0
    V = AF * 7.0
    A_ENV = 553.0
    U = 8.8
    TAU_G = 0.85
    RHO_AIR = 1.2
    CP_AIR = 1005.0
    LAMBDA_V = 2.45e6
    HEATER_KP = 194000.0
    HEATER_PMAX = 220000.0
    VENT_TRIGGER_OFFSET = 2.0

    dt = 3600.0 / sub_steps
    Tin_h = np.empty(n)
    RHin_h = np.empty(n)
    Tin = Tout[0]
    rho_v_in = rho_v_out[0]

    for i in range(n):
        Ti_acc = 0.0
        RH_acc = 0.0
        To = Tout[i]
        Gi = G[i]
        Pi = P_kpa[i]
        h = hour_local[i]
        rvo = rho_v_out[i]
        if h >= 6 and h < 18:
            sp = 20.0
        elif h >= 18:
            sp = 16.0
        else:
            sp = 18.0

        for s in range(sub_steps):
            err = sp - Tin
            Qheater = HEATER_KP * err if err > 0.0 else 0.0
            if Qheater > HEATER_PMAX:
                Qheater = HEATER_PMAX

            if Tin > sp + VENT_TRIGGER_OFFSET:
                ach = ach_open
                u_fan = 1.0
            else:
                ach = ach_closed
                u_fan = 0.0
            Vdot = ach * V / 3600.0
            mdot_vent = RHO_AIR * Vdot
            Qvent = mdot_vent * CP_AIR * (Tin - To)

            Qsolar = TAU_G * Gi * AF
            Qcond = U * A_ENV * (Tin - To)

            es_in = 0.6108 * np.exp(17.27 * Tin / (Tin + 237.3))
            Delta = 4098.0 * es_in / ((Tin + 237.3) ** 2)
            gamma = 0.000665 * Pi
            VP_in = rho_v_in * 1000.0 * (Tin + 273.15) / 2167.0
            VPD_air = es_in - VP_in
            if VPD_air < 0.0:
                VPD_air = 0.0

            Rn = f_net_rad * TAU_G * Gi

            if transpiration_mode == 1:
                Rc = TAU_G * Gi
                rs = rs_min + (rs_dark - rs_min) * np.exp(-Rc / rs_r0)
                LE = (Delta * Rn + RHO_AIR * CP_AIR * VPD_air / ra) / \
                     (Delta + gamma * (1.0 + rs / ra))
            elif transpiration_mode == 2:
                u_eff_local = u_min_fao + u_fan * (u_max_fao - u_min_fao)
                ra_fao = 208.0 / u_eff_local
                LE = (Delta * Rn + RHO_AIR * CP_AIR * VPD_air / ra_fao) / \
                     (Delta + gamma * (1.0 + rs_fao / ra_fao))
            else:
                LE = alpha_pt * (Delta / (Delta + gamma)) * Rn

            if LE < 0.0:
                LE = 0.0
            Qtrans = LE * AF
            if transpiration_mode == 0:
                Qtrans += trans_floor
            mdot_trans = Qtrans / LAMBDA_V

            dTin = (dt / (RHO_AIR * CP_AIR * V)) * (Qsolar + Qheater - Qcond - Qvent - Qtrans)
            Tin_next = Tin + dTin
            drho = (dt / V) * (mdot_trans - Vdot * (rho_v_in - rvo))
            rho_v_next = rho_v_in + drho
            if rho_v_next < 0.0:
                rho_v_next = 0.0
            es_next = 0.6108 * np.exp(17.27 * Tin_next / (Tin_next + 237.3))
            rho_v_sat = (2167.0 * es_next / (Tin_next + 273.15)) / 1000.0
            if rho_v_next > rho_v_sat:
                rho_v_next = rho_v_sat

            Tin = Tin_next
            rho_v_in = rho_v_next
            Ti_acc += Tin
            VP_now = rho_v_in * 1000.0 * (Tin + 273.15) / 2167.0
            SVP_now = 0.6108 * np.exp(17.27 * Tin / (Tin + 237.3))
            RH_now = VP_now / SVP_now * 100.0
            if RH_now > 100.0:
                RH_now = 100.0
            RH_acc += RH_now

        Tin_h[i] = Ti_acc / sub_steps
        RHin_h[i] = RH_acc / sub_steps

    return Tin_h, RHin_h


# ------------------------------------------------------------------------------
# Forcing cache (same preprocessing as driver.run_year)
# ------------------------------------------------------------------------------
_FORCING = {}


def forcing(year):
    if year not in _FORCING:
        cfg = driver.YEARS[year]
        out = driver.load_outdoor(cfg["outdoor"])
        ind = driver.load_indoor(cfg["indoor"], cfg["time_col"])
        df = pd.merge(out, ind, on="time", how="inner").sort_values("time").reset_index(drop=True)
        Tout = df["temperature_2m"].to_numpy()
        G = np.clip(df["shortwave_radiation"].to_numpy(), 0.0, None)
        P_kpa = df["surface_pressure"].to_numpy() / 10.0
        hour_local = df["time"].dt.hour.to_numpy()
        RH_out = df["relative_humidity_2m"].to_numpy()
        VP_out = RH_out / 100.0 * driver.svp_kpa(Tout)
        rho_v_out = 2167.0 * VP_out / (Tout + 273.15) / 1000.0
        _FORCING[year] = (df, Tout, G, P_kpa, hour_local, rho_v_out)
    return _FORCING[year]


def run(year, p):
    """Simulate one year with config p; returns df (24 h spin-up dropped)."""
    df, Tout, G, P_kpa, hour_local, rho_v_out = forcing(year)
    q = full_params(p)
    Tin, RH = simulate_ext(len(df), SUB_STEPS, Tout, G, P_kpa, hour_local, rho_v_out,
                           q["mode"], q["ach_closed"], q["ach_open"], q["ra"], q["rs_min"],
                           q["rs_dark"], q["rs_r0"], q["rs_fao"], q["u_min_fao"], q["u_max_fao"],
                           q["trans_floor"], q["alpha_pt"], q["f_net_rad"])
    out = df[["time", "T_obs", "RH_obs", "shortwave_radiation"]].copy()
    out["T_sim"] = Tin
    out["RH_sim"] = RH
    out = out[out["time"] >= out["time"].iloc[0] + pd.Timedelta(hours=24)].reset_index(drop=True)
    out["VPD_obs"] = vpd(out["T_obs"].to_numpy(), out["RH_obs"].to_numpy())
    out["VPD_sim"] = vpd(out["T_sim"].to_numpy(), out["RH_sim"].to_numpy())
    return out


def window(df, year, which):
    if which == "heldout":
        return df[df["time"] >= CAL_END] if year == 2019 else df
    if which == "calibration":
        return df[df["time"] < CAL_END] if year == 2019 else df.iloc[0:0]
    return df  # "full"


# ------------------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------------------
def vpd(T, RH):
    return driver.svp_kpa(T) * (1.0 - RH / 100.0)


def metrics(obs, sim):
    k = ~(np.isnan(obs) | np.isnan(sim))
    o, s = obs[k], sim[k]
    e = s - o
    return dict(rmse=float(np.sqrt(np.mean(e ** 2))), mae=float(np.mean(np.abs(e))),
                bias=float(np.mean(e)),
                ef=float(1.0 - np.sum(e ** 2) / np.sum((o - o.mean()) ** 2)), n=int(k.sum()))


def score(df):
    """Metrics for T, RH, VPD on one dataframe -> flat dict."""
    r = {}
    for v in ["T", "RH", "VPD"]:
        for k, x in metrics(df[f"{v}_obs"].to_numpy(), df[f"{v}_sim"].to_numpy()).items():
            r[f"{v}_{k}"] = x
    return r


def heldout_scores(p):
    """Per-year held-out metrics and their mean (the paper's averaging)."""
    rows = []
    for y in YEARS:
        r = score(window(run(y, p), y, "heldout"))
        r["year"] = y
        rows.append(r)
    per_year = pd.DataFrame(rows)
    return per_year, per_year.drop(columns=["year"]).mean()


def check_reproduction():
    """simulate_ext must equal model.simulate at the original settings."""
    worst = 0.0
    for name, p in CONFIGS.items():
        q = full_params(p)
        df, Tout, G, P_kpa, hour_local, rho_v_out = forcing(2018)
        a = model.simulate(len(df), SUB_STEPS, Tout, G, P_kpa, hour_local, rho_v_out,
                           q["mode"], q["ach_closed"], q["ach_open"], q["ra"], q["rs_min"],
                           q["rs_dark"], q["rs_r0"], q["rs_fao"], q["u_min_fao"], q["u_max_fao"],
                           q["trans_floor"])
        b = simulate_ext(len(df), SUB_STEPS, Tout, G, P_kpa, hour_local, rho_v_out,
                         q["mode"], q["ach_closed"], q["ach_open"], q["ra"], q["rs_min"],
                         q["rs_dark"], q["rs_r0"], q["rs_fao"], q["u_min_fao"], q["u_max_fao"],
                         q["trans_floor"], q["alpha_pt"], q["f_net_rad"])
        d = max(np.nanmax(np.abs(a[0] - b[0])), np.nanmax(np.abs(a[1] - b[1])))
        worst = max(worst, d)
        print(f"  {name:<34} max |diff| = {d:.2e}")
    assert worst < 1e-9, "simulate_ext does not reproduce model.simulate"
    print("  simulate_ext reproduces model.simulate exactly.")


if __name__ == "__main__":
    print("Checking simulate_ext against model.simulate (2018):")
    check_reproduction()
    print("\nHeld-out scores (mean of per-year metrics):")
    rows = {}
    for name, p in CONFIGS.items():
        rows[name] = heldout_scores(p)[1]
    tab = pd.DataFrame(rows).T[["T_rmse", "T_ef", "RH_rmse", "RH_ef", "VPD_rmse", "VPD_ef"]]
    print(tab.round(2).to_string())
    tab.to_csv(os.path.join(OUT_DIR, "heldout_baseline.csv"))
