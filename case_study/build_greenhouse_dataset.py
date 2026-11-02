#!/usr/bin/env python3
"""
================================================================================
GREENHOUSE DIGITAL-TWIN CLIMATE DATASET — full pipeline, single file
================================================================================
Source data : Open-Meteo Historical Weather API, hourly
Location    : SEECS, NUST, Islamabad (33.6456N, 72.9903E), Asia/Karachi (UTC+5)
Input       : data/openmeteo_islamabad_hourly.csv  (17,544 hourly rows)
Output      : outputs/greenhouse_digital_twin_1min.csv / .parquet (1,052,581 rows)

This script builds the case-study dataset in order: Stage 1-2 (load + 1-minute interpolation), Stage 3-5
(outdoor humidity-derived + PPFD/DLI variables), Stage 6-8 (coupled indoor
energy/moisture balance ODE + final indoor variables), and final export.

Two real bugs were found and fixed during development; both are implemented
correctly here and flagged in-place with "BUG FIX" comments so you can see
exactly what changed and why:

  BUG FIX #1 — the absolute-humidity constant 216.7 is calibrated for vapor
  pressure in hPa, but this pipeline computes vapor pressure in kPa
  everywhere (FAO-56 Magnus formula convention). Using 216.7 directly against
  a kPa value under-reports humidity by exactly 10x. Fixed by using 2167
  (=216.7*10) wherever the AH/VP conversion is applied to a kPa value.
  Verified from scratch against the ideal gas law: AH[g/m3] = e[Pa]*Mw/(R*T).

  BUG FIX #2 — once bug #1 was fixed, the indoor moisture balance (Stage 7)
  diverged to physically impossible values during long low-ventilation
  stretches (58% of this record has u_fan<0.05, one stretch ~2200h), because
  the given mass balance has no moisture sink other than mechanical
  ventilation. Fixed by adding a condensation closure: once predicted indoor
  humidity would exceed saturation at the current indoor temperature, the
  excess is capped there (instant condensation onto the cool glazing/canopy).
  This is a physical closure added on top of the source document's
  equations, not something the source specifies -- flagged clearly since
  it's a modeling judgment call, not a pure units fix.

There is also one documented, unavoidable ASSUMPTION (not a bug): the
sinusoidal solar-radiation interpolation's amplitude parameter "A" is only
described qualitatively in its source paper (Balog et al. 2023, Sustainability
15(3):2762) -- their actual calibration requires real 1-minute ground-truth
measurements, which don't exist for this site. See the "ASSUMPTION" comment
in solar_downscale() for the stand-in formula used and why.

Requires: numpy, pandas, scipy, numba, pyarrow
    pip install numpy pandas scipy numba pyarrow --break-system-packages
================================================================================
"""
import os
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from numba import njit

# ==============================================================================
# CONFIG
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(SCRIPT_DIR, 'data', 'openmeteo_islamabad_hourly.csv')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'outputs')

LAT = 33.6456      # deg N
LON = 72.9903      # deg E
TZ_OFFSET = 5.0    # Asia/Karachi, UTC+5 (standard meridian = 75 E)
ISC = 1367.0       # W/m2, solar constant

TAU_G = 0.80        # glazing transmittance, 6mm twin-wall polycarbonate

# Greenhouse geometry / energy-balance constants (case-study greenhouse specification)
AF = 30.0                                     # floor area, m2 (5x6 m)
A_ENV = 87.64                                 # total envelope conductive area, m2
U = 3.5                                       # overall heat-transfer coeff, W/m2K
V = 30.0*2.1 + 30.0*(3.5-2.1)/2.0             # air volume, m3: rect prism (eave
                                               # height 2.1m) + gable prism (ridge
                                               # 3.5m), derived from the given dims
RHO_AIR = 1.2                                 # kg/m3
CP_AIR = 1005.0                               # J/kg K
PH = 2800.0                                   # heater rated power, W (Phoenix heater)
PLED = 1000.0                                 # LED rated power, W (Lucius LED)
LAMBDA_V = 2.45e6                             # J/kg, latent heat of vaporization
ALPHA_PT = 1.26                               # Priestley-Taylor coefficient
VDOT_MAX = 2.36                               # m3/s, max ventilation (2x16in fans, 2500 CFM ea.)
DT = 60.0                                     # s, Forward Euler step (1 minute)

# Thermostat control setpoints (a modelling choice; the greenhouse
# specification does not define a control strategy)
HEAT_ON_T, HEAT_OFF_T = 16.0, 20.0            # C: heater full power below, off above
FAN_OFF_T, FAN_ON_T = 26.0, 30.0              # C: fan off below, full power above
LED_ON_HOUR, LED_OFF_HOUR = 6, 22             # local clock hour, fixed photoperiod


# ==============================================================================
# SHARED FORMULAS (FAO-56 / Magnus-Tetens, used in multiple stages)
# ==============================================================================
def svp_kpa(Tc):
    """Saturation vapor pressure, kPa, T in Celsius. FAO-56 Magnus formula."""
    return 0.6108*np.exp(17.27*Tc/(Tc+237.3))


def dewpoint_from_RH_T(Tc, RHpct):
    """Dew point, C, via Magnus inversion. RH in %, T in Celsius."""
    RHpct = np.clip(RHpct, 0.1, 100.0)
    gamma = np.log(RHpct/100.0) + 17.27*Tc/(237.3+Tc)
    return 237.3*gamma/(17.27-gamma)


# ==============================================================================
# STAGE 1-2: load hourly data, interpolate to 1-minute resolution
# ==============================================================================
def solar_geometry(dt_index):
    """cos(zenith angle) and extraterrestrial normal irradiance I_on for an
    array of naive local timestamps (Asia/Karachi standard time). Standard
    Duffie & Beckman solar-position equations -- not from the Balog paper,
    which treats these as already-known inputs."""
    dti = pd.DatetimeIndex(dt_index)
    doy = dti.dayofyear.to_numpy().astype(float)
    hour_local = (dti.hour.to_numpy().astype(float)
                  + dti.minute.to_numpy().astype(float)/60.0
                  + dti.second.to_numpy().astype(float)/3600.0)

    B = 2*np.pi*(doy-1)/365.0
    EoT = 229.2*(0.000075 + 0.001868*np.cos(B) - 0.032077*np.sin(B)
                 - 0.014615*np.cos(2*B) - 0.04089*np.sin(2*B))   # minutes
    Lstd = 15.0*TZ_OFFSET
    solar_time = hour_local + (4.0*(Lstd - LON) + EoT)/60.0       # hours
    omega = np.deg2rad(15.0*(solar_time - 12.0))                   # hour angle, rad

    decl = np.deg2rad(23.45*np.sin(np.deg2rad(360.0*(284.0+doy)/365.0)))
    lat = np.deg2rad(LAT)

    cos_z = np.sin(lat)*np.sin(decl) + np.cos(lat)*np.cos(decl)*np.cos(omega)
    cos_z = np.clip(cos_z, 0.0, 1.0)

    E0 = 1.0 + 0.033*np.cos(np.deg2rad(360.0*doy/365.0))   # eccentricity correction
    I_on = ISC*E0
    return cos_z, I_on


def solar_downscale(df, minute_idx, hour_epoch_min, min_epoch_min,
                     cosz_h, Ion_h, cosz_m, Ion_m, rad_col):
    """Sinusoidal-physical downscaling of one solar radiation column to
    1-minute resolution, following Balog et al. (2023), Sustainability
    15(3):2762, Eq. 4-6 (verified against the paper's full text):

        S_h  = I_h / (I_on * cos(theta_z,h)^1.15)                    (Eq 5)
        S_i  = S1 + (S2-S1)/(H2-H1)*(i-H1) + A*sin(2pi/(H2-H1)*(i-H1))  (Eq 6)
        I_i  = I_on,i * cos(theta_z,i)^1.15 * S_i                     (Eq 4, inverted)
    """
    I_h = df[rad_col].to_numpy()
    Io_h = Ion_h*cosz_h
    # Sh = Ih/(Ion*cosz^1.15) = Ih/(Ion*cosz*cosz^0.15) = Ih/(Io*cosz^0.15)
    # (mathematically identical to Eq 5, just factored differently)
    S_h = np.where(cosz_h > 0.02,
                   I_h/np.maximum(Io_h*np.power(np.maximum(cosz_h, 1e-6), 0.15), 1e-9),
                   0.0)

    # ASSUMPTION -- amplitude A: the paper defines A only qualitatively
    # ("~0 when clearness index kc>0.7", "max observed A=0.08"), calibrated
    # against real 1-minute ground-truth pyranometer data the authors had at
    # their site (Rome, Italy, 2018). We have no such minute-truth data for
    # Islamabad, so the paper's actual calibration procedure cannot be
    # reproduced from hourly data alone. Stand-in used here, confirmed
    # consistent with the paper's own daily clearness classification
    # (kc<0.65 cloudy / kc>=0.65 clear, their Table 2):
    #     A_day = 0.08 * (1 - kc_day)   for kc_day < 0.65 (cloudy)
    #     A_day = 0                     for kc_day >= 0.65 (clear)
    # Applied independently to shortwave/direct/diffuse using each column's
    # own kc_day -- the paper only demonstrates GHI; extending the same
    # Bourges-derived formula to direct/diffuse is an extrapolation beyond
    # what the paper validates.
    Io_h_all = Ion_h*cosz_h
    tmp = pd.DataFrame({'day': df['t'].dt.date, 'I': I_h, 'Io': Io_h_all})
    g = tmp.groupby('day').sum()
    kc = (g['I'] / g['Io'].replace(0, np.nan)).clip(lower=0, upper=1.2)
    kc_day = kc.reindex(tmp['day']).to_numpy()
    A_day = np.where(kc_day >= 0.65, 0.0, 0.08*(1.0-kc_day))
    A_day = np.nan_to_num(A_day, nan=0.0)

    # bucket each minute into its enclosing hour interval [H1,H2)
    hour_bin = np.searchsorted(hour_epoch_min, min_epoch_min, side='right') - 1
    hour_bin = np.clip(hour_bin, 0, len(df)-2)   # last hour has no H2 -> held flat below

    H1 = hour_epoch_min[hour_bin]
    H2 = hour_epoch_min[hour_bin+1]
    S1 = S_h[hour_bin]
    S2 = S_h[hour_bin+1]
    A = A_day[hour_bin]
    frac = (min_epoch_min - H1)
    span = (H2 - H1)
    S_i = S1 + (S2-S1)/span*frac + A*np.sin(2*np.pi/span*frac)

    last_mask = min_epoch_min >= hour_epoch_min[-1]
    S_i = np.where(last_mask, S_h[-1], S_i)   # hold last hourly value flat past the end

    Io_m = Ion_m*cosz_m
    I_i = Io_m*np.power(np.maximum(cosz_m, 1e-6), 0.15)*S_i
    I_i = np.clip(I_i, 0.0, None)
    I_i = np.where(cosz_m <= 0.001, 0.0, I_i)   # force night to exactly zero
    return I_i


def run_stage1_2():
    print("=== Stage 1-2: load hourly data, interpolate to 1-minute ===")
    df = pd.read_csv(INPUT_CSV)
    df['t'] = pd.to_datetime(df['time'])   # accepts the bundled CSV and fresh fetch_open_meteo.py output
    df = df.sort_values('t').reset_index(drop=True)
    assert (df['t'].diff().dropna() == pd.Timedelta(hours=1)).all(), "gap in hourly series!"

    t0, t1 = df['t'].iloc[0], df['t'].iloc[-1]
    minute_idx = pd.date_range(t0, t1, freq='1min')
    print("hourly rows:", len(df), " minute rows:", len(minute_idx))

    hour_epoch_min = (df['t'] - t0).dt.total_seconds().to_numpy() / 60.0
    min_epoch_min = (minute_idx - t0).total_seconds().to_numpy() / 60.0

    cosz_h, Ion_h = solar_geometry(df['t'])
    cosz_m, Ion_m = solar_geometry(minute_idx)

    # Temperature & humidity family: PCHIP (monotone shape-preserving cubic
    # Hermite interpolation) -- matches the source paper's method for T/humidity.
    def pchip_col(col):
        f = PchipInterpolator(hour_epoch_min, df[col].to_numpy())
        return f(min_epoch_min)

    T_m = pchip_col('temperature_2m')
    RH_m = pchip_col('relative_humidity_2m')
    Tdew_raw_m = pchip_col('dew_point_2m')
    P_m = pchip_col('surface_pressure')   # hPa

    # Solar radiation family: sinusoidal-physical model (see solar_downscale above)
    GHI_m = solar_downscale(df, minute_idx, hour_epoch_min, min_epoch_min,
                             cosz_h, Ion_h, cosz_m, Ion_m, 'shortwave_radiation')
    DIR_m = solar_downscale(df, minute_idx, hour_epoch_min, min_epoch_min,
                             cosz_h, Ion_h, cosz_m, Ion_m, 'direct_radiation')
    DIF_m = solar_downscale(df, minute_idx, hour_epoch_min, min_epoch_min,
                             cosz_h, Ion_h, cosz_m, Ion_m, 'diffuse_radiation')

    out = pd.DataFrame({
        'time': minute_idx,
        'temperature_2m': T_m,
        'relative_humidity_2m': RH_m,
        'dew_point_2m_raw': Tdew_raw_m,
        'shortwave_radiation': GHI_m,
        'direct_radiation': DIR_m,
        'diffuse_radiation': DIF_m,
        'surface_pressure': P_m,
    })
    print("NaNs:", out.isna().sum().sum(),
          " GHI range:", out['shortwave_radiation'].min(), "-", out['shortwave_radiation'].max())
    return out


# ==============================================================================
# STAGE 3-5: outdoor humidity/vapor-pressure vars, PPFD/DLI outdoor+indoor
# ==============================================================================
def run_stage3_5(df):
    print("=== Stage 3-5: humidity-derived variables, PPFD/DLI ===")
    T = df['temperature_2m'].to_numpy()
    RH = df['relative_humidity_2m'].to_numpy()
    Rs = df['shortwave_radiation'].to_numpy()

    SVP = svp_kpa(T)
    VP = (RH/100.0)*SVP
    VPD = SVP - VP
    Tdew = dewpoint_from_RH_T(T, RH)

    # BUG FIX #1 -- absolute humidity unit constant.
    # AH = 216.7 * e / T is the standard ideal-gas-law absolute-humidity
    # formula, but that constant is calibrated for e in hPa (confirmed
    # against the source doc's own derivation note, and independently
    # against a from-scratch ideal-gas-law calc:
    #     AH[g/m3] = e[Pa]*Mw/(R*T), Mw=18.015 g/mol, R=8.314 J/mol/K
    #     -> constant 216.7 pairs with e in hPa).
    # VP here is in kPa (FAO-56 Magnus, 0.6108 coefficient), so the paired
    # constant is 216.7*10 = 2167, not 216.7. Using 216.7 directly against a
    # kPa value would under-report absolute humidity by exactly 10x -- this
    # was the original bug, now fixed.
    AH = 2167.0*VP/(T+273.15)   # g/m3

    dT = T - Tdew
    cond_risk = np.where(dT > 3, 'Low', np.where(dT >= 1, 'Moderate', 'High'))

    # Stage 4: outdoor PPFD / DLI (Thimijan & Heins conversion, 1 W/m2 ~ 2.02 umol/m2/s PAR)
    PPFD_out = 2.02*Rs
    # Stage 5: indoor PPFD / DLI (glazing transmittance)
    PPFD_in = TAU_G*PPFD_out

    df = df.copy()
    df['_day'] = df['time'].dt.date
    df['SVP_out_kPa'] = SVP
    df['VP_out_kPa'] = VP
    df['VPD_out_kPa'] = VPD
    df['dew_point_2m_calc'] = Tdew
    df['AH_out_gm3'] = AH
    df['condensation_risk_outdoor'] = cond_risk
    df['PPFD_out_umol_m2s'] = PPFD_out
    df['DLI_out_daily_mol_m2'] = df.groupby('_day')['PPFD_out_umol_m2s'].transform(lambda s: (s*60/1e6).cumsum())
    df['PPFD_in_umol_m2s'] = PPFD_in
    df['DLI_in_daily_mol_m2'] = df.groupby('_day')['PPFD_in_umol_m2s'].transform(lambda s: (s*60/1e6).cumsum())
    df = df.drop(columns=['_day'])

    print("AH_out_gm3 range:", df['AH_out_gm3'].min(), "-", df['AH_out_gm3'].max(),
          " mean:", df['AH_out_gm3'].mean())
    return df


# ==============================================================================
# STAGE 6-8: coupled indoor energy + moisture balance, final indoor variables
# ==============================================================================
@njit(cache=True)
def _svp_kpa_scalar(Tc):
    return 0.6108*np.exp(17.27*Tc/(Tc+237.3))


@njit(cache=True)
def _simulate(n, Tout, G, P_kpa, hour_local, rho_v_out,
              AF, A_ENV, U, TAU_G, V, RHO_AIR, CP_AIR, PH, PLED,
              LAMBDA_V, ALPHA_PT, VDOT_MAX, DT,
              HEAT_ON_T, HEAT_OFF_T, FAN_OFF_T, FAN_ON_T, LED_ON_HOUR, LED_OFF_HOUR):
    Tin = np.empty(n)
    rho_v_in = np.empty(n)
    u_h_arr = np.empty(n)
    u_led_arr = np.empty(n)
    u_fan_arr = np.empty(n)
    Qsolar_a = np.empty(n); Qheater_a = np.empty(n); QLED_a = np.empty(n)
    Qcond_a = np.empty(n); Qvent_a = np.empty(n); Qtrans_a = np.empty(n)

    # Initial condition: indoor = outdoor at t=0 (equilibrium start). First
    # 24h of output is flagged spin_up_flag=True downstream for the control
    # loop transient to settle; discard those rows if you want steady-state-
    # only analysis.
    Tin[0] = Tout[0]
    rho_v_in[0] = rho_v_out[0]

    for i in range(n):
        Ti = Tin[i]

        # ---- thermostat controls (linear ramp between on/off setpoints) ----
        if Ti <= HEAT_ON_T:
            u_h = 1.0
        elif Ti >= HEAT_OFF_T:
            u_h = 0.0
        else:
            u_h = (HEAT_OFF_T - Ti)/(HEAT_OFF_T - HEAT_ON_T)

        if Ti <= FAN_OFF_T:
            u_fan = 0.0
        elif Ti >= FAN_ON_T:
            u_fan = 1.0
        else:
            u_fan = (Ti - FAN_OFF_T)/(FAN_ON_T - FAN_OFF_T)

        h = hour_local[i]
        u_led = 1.0 if (h >= LED_ON_HOUR and h < LED_OFF_HOUR) else 0.0

        u_h_arr[i] = u_h; u_led_arr[i] = u_led; u_fan_arr[i] = u_fan

        Qsolar = TAU_G*G[i]*AF
        Qheater = PH*u_h
        QLED = PLED*u_led
        Qcond = U*A_ENV*(Ti - Tout[i])
        Vdot = u_fan*VDOT_MAX
        mdot_vent = RHO_AIR*Vdot
        Qvent = mdot_vent*CP_AIR*(Ti - Tout[i])

        es_Tin = _svp_kpa_scalar(Ti)
        Delta = 4098.0*es_Tin/((Ti+237.3)**2)
        gamma_psy = 0.000665*P_kpa[i]
        Rn = TAU_G*G[i]
        Qtrans = ALPHA_PT*(Delta/(Delta+gamma_psy))*Rn*AF
        Qtrans = max(Qtrans, 0.0)   # no negative transpiration (no dew-formation term modeled here)

        Qsolar_a[i] = Qsolar; Qheater_a[i] = Qheater; QLED_a[i] = QLED
        Qcond_a[i] = Qcond; Qvent_a[i] = Qvent; Qtrans_a[i] = Qtrans

        mdot_trans = Qtrans/LAMBDA_V

        if i < n-1:
            dTin = (DT/(RHO_AIR*CP_AIR*V))*(Qsolar+Qheater+QLED-Qcond-Qvent-Qtrans)
            Tin[i+1] = Ti + dTin

            drho = (DT/V)*(mdot_trans - Vdot*(rho_v_in[i]-rho_v_out[i]))
            rho_v_next = rho_v_in[i] + drho
            if rho_v_next < 0.0:
                rho_v_next = 0.0

            # BUG FIX #2 -- condensation closure (physical addition, not in
            # the source document's equations). The given mass balance has
            # no moisture sink other than ventilation, so during long
            # low-ventilation stretches (58% of this record has u_fan<0.05,
            # one continuous run ~2200h) the raw ODE accumulates moisture
            # without bound -- not a numerical instability, but the missing
            # physical process of condensation onto the cool glazing/canopy
            # once air reaches saturation. Fixed by capping rho_v at the
            # saturation density for the new indoor temperature each step.
            es_next = _svp_kpa_scalar(Tin[i+1])
            rho_v_sat_gm3 = 2167.0*es_next/(Tin[i+1]+273.15)   # BUG FIX #1's constant, reused here
            rho_v_sat = rho_v_sat_gm3/1000.0   # convert g/m3 -> kg/m3 to match rho_v_in's units
            if rho_v_next > rho_v_sat:
                rho_v_next = rho_v_sat

            rho_v_in[i+1] = rho_v_next

    return Tin, rho_v_in, u_h_arr, u_led_arr, u_fan_arr, Qsolar_a, Qheater_a, QLED_a, Qcond_a, Qvent_a, Qtrans_a


def run_stage6_8(df):
    print("=== Stage 6-8: coupled indoor energy/moisture balance ===")
    print("Greenhouse air volume V =", V, "m3")
    n = len(df)

    Tout = df['temperature_2m'].to_numpy()
    G = df['shortwave_radiation'].to_numpy()
    P_kpa = df['surface_pressure'].to_numpy()/10.0   # hPa -> kPa (FAO-56 gamma needs kPa)
    hour_local = df['time'].dt.hour.to_numpy()
    rho_v_out = df['AH_out_gm3'].to_numpy()/1000.0    # g/m3 -> kg/m3

    (Tin, rho_v_in, u_h, u_led, u_fan,
     Qsolar, Qheater, QLED, Qcond, Qvent, Qtrans) = _simulate(
        n, Tout, G, P_kpa, hour_local, rho_v_out,
        AF, A_ENV, U, TAU_G, V, RHO_AIR, CP_AIR, PH, PLED,
        LAMBDA_V, ALPHA_PT, VDOT_MAX, DT,
        HEAT_ON_T, HEAT_OFF_T, FAN_OFF_T, FAN_ON_T, LED_ON_HOUR, LED_OFF_HOUR)

    # ---- Stage 8: final indoor climate variables ----
    rho_v_in_gm3 = rho_v_in*1000.0
    # Same fix as BUG FIX #1: 216.7 pairs with hPa, everything else (SVP_in
    # below) is in kPa, so use 2167 to get VP_in directly in kPa.
    VP_in = rho_v_in_gm3*(Tin+273.15)/2167.0
    SVP_in = svp_kpa(Tin)
    RH_in = np.clip(VP_in/SVP_in*100.0, 0.0, 100.0)
    VPD_in = SVP_in - VP_in
    ln_ratio = np.log(np.clip(VP_in/0.6108, 1e-6, None))
    Tdew_in = 237.3*ln_ratio/(17.27-ln_ratio)
    dT_in = Tin - Tdew_in
    cond_risk_in = np.where(dT_in > 3, 'Low', np.where(dT_in >= 1, 'Moderate', 'High'))

    df = df.copy()
    df['u_heater'] = u_h
    df['u_LED'] = u_led
    df['u_fan'] = u_fan
    df['Q_solar_W'] = Qsolar
    df['Q_heater_W'] = Qheater
    df['Q_LED_W'] = QLED
    df['Q_cond_W'] = Qcond
    df['Q_vent_W'] = Qvent
    df['Q_trans_W'] = Qtrans
    df['T_indoor_C'] = Tin
    df['rho_v_indoor_gm3'] = rho_v_in_gm3
    df['VP_indoor_kPa'] = VP_in
    df['SVP_indoor_kPa'] = SVP_in
    df['RH_indoor_pct'] = RH_in
    df['VPD_indoor_kPa'] = VPD_in
    df['dew_point_indoor_C'] = Tdew_in
    df['condensation_risk_indoor'] = cond_risk_in
    df['spin_up_flag'] = df['time'] < (df['time'].iloc[0] + pd.Timedelta(hours=24))

    print("T_indoor range:", Tin.min(), "-", Tin.max())
    print("RH_indoor range:", RH_in.min(), "-", RH_in.max())
    print("rho_v_indoor_gm3 range:", df['rho_v_indoor_gm3'].min(), "-", df['rho_v_indoor_gm3'].max())
    return df


# ==============================================================================
# FINAL ASSEMBLY & EXPORT
# ==============================================================================
def export_final(df):
    print("=== Final assembly & export ===")
    cols = [
        'time',
        'temperature_2m', 'relative_humidity_2m', 'dew_point_2m_raw',
        'shortwave_radiation', 'direct_radiation', 'diffuse_radiation', 'surface_pressure',
        'SVP_out_kPa', 'VP_out_kPa', 'VPD_out_kPa', 'dew_point_2m_calc',
        'AH_out_gm3', 'condensation_risk_outdoor',
        'PPFD_out_umol_m2s', 'DLI_out_daily_mol_m2', 'PPFD_in_umol_m2s', 'DLI_in_daily_mol_m2',
        'u_heater', 'u_LED', 'u_fan',
        'Q_solar_W', 'Q_heater_W', 'Q_LED_W', 'Q_cond_W', 'Q_vent_W', 'Q_trans_W',
        'T_indoor_C', 'rho_v_indoor_gm3', 'VP_indoor_kPa', 'SVP_indoor_kPa',
        'RH_indoor_pct', 'VPD_indoor_kPa', 'dew_point_indoor_C', 'condensation_risk_indoor',
        'spin_up_flag',
    ]
    df = df[cols]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, 'greenhouse_digital_twin_1min.csv')
    parquet_path = os.path.join(OUTPUT_DIR, 'greenhouse_digital_twin_1min.parquet')
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path)

    print(df.shape, "NaNs:", df.isna().sum().sum())
    print("Wrote:", csv_path)
    print("Wrote:", parquet_path)
    return df


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == '__main__':
    df = run_stage1_2()
    df = run_stage3_5(df)
    df = run_stage6_8(df)
    df = export_final(df)
    print("\nPipeline complete.", len(df), "rows,", len(df.columns), "columns.")
