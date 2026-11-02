import json
import os
import numpy as np
import pandas as pd
import model
from model import simulate

# All data paths are relative to this script's own folder, so the project can
# be dropped anywhere (e.g. cloned into Antigravity / any local IDE) without
# editing paths. Override with the GREENHOUSE_DATA_DIR env var if you'd
# rather point at a data folder living somewhere else.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("GREENHOUSE_DATA_DIR", os.path.join(SCRIPT_DIR, "data"))

BASE = os.path.join(DATA_DIR, "hoogstraten")

YEARS = {
    2018: dict(outdoor=os.path.join(DATA_DIR, "openmeteo_2018.json"),
               indoor=os.path.join(BASE, "Environment", "T&RH", "2018", "CENTRAL.xlsx"),
               time_col="Timestep"),
    2019: dict(outdoor=os.path.join(DATA_DIR, "openmeteo_2019.json"),
               indoor=os.path.join(BASE, "Environment", "T&RH", "2019", "CENTRAL.xlsx"),
               time_col="TimeStep"),
    2020: dict(outdoor=os.path.join(DATA_DIR, "openmeteo_2020.json"),
               indoor=os.path.join(BASE, "Environment", "T&RH", "2020", "CENTRAL.xlsx"),
               time_col="Time"),
}


def load_outdoor(path):
    with open(path) as f:
        d = json.load(f)
    h = d["hourly"]
    df = pd.DataFrame({
        "time": pd.to_datetime(h["time"]),
        "temperature_2m": h["temperature_2m"],
        "shortwave_radiation": h["shortwave_radiation"],
        "surface_pressure": h["surface_pressure"],   # hPa
        "relative_humidity_2m": h["relative_humidity_2m"],
    })
    return df


def svp_kpa(Tc):
    return 0.6108 * np.exp(17.27 * Tc / (Tc + 237.3))


def load_indoor(path, time_col):
    df = pd.read_excel(path)
    cols = df.columns.tolist()
    temp_col = [c for c in cols if c.lower() in ("temperature", "t")][0]
    rh_col = [c for c in cols if c.lower() in ("relativehumidity", "rh")][0]
    df = df.rename(columns={time_col: "time", temp_col: "T_obs", rh_col: "RH_obs"})
    df = df.set_index("time")[["T_obs", "RH_obs"]].resample("1h").mean().reset_index()
    return df


def metrics(obs, sim):
    mask = ~(np.isnan(obs) | np.isnan(sim))
    o, s = obs[mask], sim[mask]
    err = s - o
    rmse = np.sqrt(np.mean(err ** 2))
    mae = np.mean(np.abs(err))
    ef = 1.0 - np.sum(err ** 2) / np.sum((o - np.mean(o)) ** 2)
    return rmse, mae, ef, err, o


def run_year(year, transpiration_mode, sub_steps=120,
             ach_closed=model.ACH_CLOSED, ach_open=model.ACH_OPEN,
             ra=model.RA, rs_min=model.RS_MIN, rs_dark=model.RS_DARK, rs_r0=model.RS_R0,
             rs_fao=model.RS_FAO, u_min_fao=model.U_MIN_FAO, u_max_fao=model.U_MAX_FAO,
             trans_floor=None):
    if trans_floor is None:
        # Only Priestley-Taylor (mode 0) uses the flat transpiration floor.
        # Stanghellini (mode 1) and FAO-56 (mode 2) have VPD-driven terms
        # that provide a natural floor via the aerodynamic component.
        trans_floor = 50000.0 if transpiration_mode == 0 else 0.0
    cfg = YEARS[year]
    out = load_outdoor(cfg["outdoor"])
    ind = load_indoor(cfg["indoor"], cfg["time_col"])

    df = pd.merge(out, ind, on="time", how="inner").sort_values("time").reset_index(drop=True)
    assert len(df) > 100, f"merge produced too few rows for {year}: {len(df)}"

    Tout = df["temperature_2m"].to_numpy()
    G = np.clip(df["shortwave_radiation"].to_numpy(), 0.0, None)
    P_kpa = df["surface_pressure"].to_numpy() / 10.0
    hour_local = df["time"].dt.hour.to_numpy()

    # outdoor absolute humidity -> kg/m3 (2167 constant, kPa convention)
    RH_out = df["relative_humidity_2m"].to_numpy()
    SVP_out = svp_kpa(Tout)
    VP_out = RH_out / 100.0 * SVP_out
    AH_out_gm3 = 2167.0 * VP_out / (Tout + 273.15)
    rho_v_out = AH_out_gm3 / 1000.0

    n = len(df)
    Tin_h, RHin_h, Qtrans_h, Qheater_h, ufan_h = simulate(
        n, sub_steps, Tout, G, P_kpa, hour_local, rho_v_out, transpiration_mode,
        ach_closed, ach_open, ra, rs_min, rs_dark, rs_r0,
        rs_fao, u_min_fao, u_max_fao, trans_floor)

    df["T_sim"] = Tin_h
    df["RH_sim"] = RHin_h
    df["Q_trans"] = Qtrans_h
    df["Q_heater"] = Qheater_h
    df["u_fan"] = ufan_h

    # drop first 24h spin-up
    df = df[df["time"] >= df["time"].iloc[0] + pd.Timedelta(hours=24)].reset_index(drop=True)
    return df


def summarize(df, label):
    t_rmse, t_mae, t_ef, t_err, _ = metrics(df["T_obs"].to_numpy(), df["T_sim"].to_numpy())
    r_rmse, r_mae, r_ef, r_err, _ = metrics(df["RH_obs"].to_numpy(), df["RH_sim"].to_numpy())
    print(f"  [{label}] Temp RMSE={t_rmse:.2f}C MAE={t_mae:.2f}C EF={t_ef:.2f}   "
          f"RH RMSE={r_rmse:.2f}% MAE={r_mae:.2f}% EF={r_ef:.2f}")
    return dict(t_rmse=t_rmse, t_mae=t_mae, t_ef=t_ef, r_rmse=r_rmse, r_mae=r_mae, r_ef=r_ef,
                r_err=r_err, time=df["time"])


if __name__ == "__main__":
    results = {}
    for year in [2018, 2019, 2020]:
        print(f"\n=== {year} ===")
        print(" -- Stage 1 (net-rad-fraction Priestley-Taylor, no VPD feedback) --")
        df1 = run_year(year, transpiration_mode=0)
        r1 = summarize(df1, "stage1")

        print(" -- Stage 2 (Stanghellini-style, VPD feedback) --")
        df2 = run_year(year, transpiration_mode=1)
        r2 = summarize(df2, "stage2")

        results[year] = dict(df1=df1, df2=df2, r1=r1, r2=r2)

    print("\n\n=== Diurnal RH bias pattern (stage2, simulated - observed, by hour) ===")
    for year in [2018, 2019, 2020]:
        df2 = results[year]["df2"]
        bias_by_hour = (df2["RH_sim"] - df2["RH_obs"]).groupby(df2["time"].dt.hour).mean()
        print(f"{year}:", {int(h): round(v, 1) for h, v in bias_by_hour.items()})

    import pickle
    with open(os.path.join(SCRIPT_DIR, "results.pkl"), "wb") as f:
        pickle.dump(results, f)
    print("\nSaved results.pkl")
