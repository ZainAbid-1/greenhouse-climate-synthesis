"""
Hoogstraten greenhouse model used for the independent-site evaluation
(Section IV of the paper).

Same governing equations as the case-study pipeline (energy balance with
Laktionov et al. 2020 solar/conductive/ventilation terms, moisture balance with
a saturation/condensation cap, 2167 absolute-humidity constant for kPa); only
geometry, setpoints, controls and the transpiration term are adapted to
Hoogstraten. Three transpiration formulations are available: Priestley-Taylor,
a Stanghellini-type Penman-Monteith term with VPD feedback, and an adapted
FAO-56 Penman-Monteith term. Net radiation uses a fraction F_NET_RAD = 0.55.
"""
import numpy as np
from numba import njit

# ==============================================================================
# GEOMETRY / MATERIAL PARAMETERS -- Hoogstraten (Section 4 of the report)
# ==============================================================================
AF = 1600.0            # floor area, m2 (32m x 50m compartment)
HEIGHT = 7.0            # m
V = AF * HEIGHT         # air volume, m3 (simplified rectangular box)
A_ENV = 553.0           # envelope conductive area, m2 (rect-box approx +5% ridge)
U = 8.8                 # W/m2K, single-pane glass
TAU_G = 0.85            # glazing transmittance
RHO_AIR = 1.2           # kg/m3
CP_AIR = 1005.0         # J/kg K
LAMBDA_V = 2.45e6       # J/kg, latent heat of vaporization

# Heater: NOT published for Hoogstraten. Modeled as a proportional controller
# tracking the published setpoints, sized so the offset at peak winter heat
# loss (~97 kW at a 20C indoor-outdoor gap) is ~0.5C -- Kp = 97000/0.5.
HEATER_KP = 194000.0    # W per degree C of setpoint error
HEATER_PMAX = 220000.0  # W, generous cap so Kp is rarely the binding constraint

# Ventilation: switched air-change rate. Vents open (trigger logic per paper)
# once Tin exceeds setpoint+2C. ACH_max=15 is the value calibrated in the
# earlier iteration (grid-searched on 2019 Jan-May, held out the rest).
# ACH_CLOSED revised: 1.0 was an overly high placeholder for "vents closed"
# infiltration. Commercial Venlo-type glasshouses are typically tighter;
# 0.2-0.5 ACH is more commonly cited for closed structures. Fixed at 0.3
# (literature-typical, not fit to this dataset).
ACH_CLOSED = 0.3
ACH_OPEN = 15.0
VENT_TRIGGER_OFFSET = 2.0   # C above setpoint

# Setpoint schedule (paper: 20C day, 16C sunset-midnight, 18C rest of night).
# Exact sunrise/sunset times aren't given in the paper -- approximated with
# fixed clock hours. Flagged as an approximation, not a paper-stated fact.
DAY_START, DAY_END = 6, 18          # 06:00-18:00 -> 20C
EVENING_END = 24                    # 18:00-24:00 -> 16C
# 00:00-06:00 -> 18C

SP_DAY, SP_EVENING, SP_NIGHT = 20.0, 16.0, 18.0

# ---- Transpiration term ----
# Stage-1 fix (net radiation fraction): literature-typical range is 0.55-0.65
# (FAO-56 / greenhouse ET literature); the grid search pushed this to 0.4
# chasing an RH minimum, which fell outside the defensible range. Reverted to
# the low end of the literature range here.
F_NET_RAD = 0.55

# Stage-2 (new): Stanghellini-style Penman-Monteith term with VPD feedback,
# replacing the flat Priestley-Taylor term.
# ra: aerodynamic resistance, s/m. CALIBRATED (grid search on 2019 Jan-May
# training subset, held out Jun-Oct 2019 + all of 2018/2020 for evaluation).
# The literature-typical range (200-300 s/m) performed much worse once the
# redundant flat floor was removed (see calibration log) -- the model needed
# far more aerodynamic coupling than the literature midpoint to match this
# specific greenhouse's real RH/VPD dynamics.
#
# rs: stomatal resistance light-response. The exact original Stanghellini
# (1987) coefficients could not be verified from primary text in this
# session, so this uses a representative, monotonic light-response curve of
# the same documented qualitative shape (resistance high in the dark,
# falling to a well-lit minimum). rs_min/rs_dark below are CALIBRATED (same
# train/test split as ra, and as ACH_max earlier in the project) -- not
# literature values. The literature-typical starting guesses (rs_min=100,
# rs_dark=400) are kept here only as a comment for reference/reproducibility.
#   literature guess: rs_min=100, rs_dark=400 -> Temp RMSE 2.2-2.5C, RH RMSE 17-19%, RH EF negative
#   calibrated:        rs_min=10,  rs_dark=350 -> Temp RMSE 1.5-1.8C, RH RMSE 8.7-9.4%, RH EF +0.17 to +0.25
RA = 20.0
RS_MIN = 10.0      # s/m, well-lit stomatal resistance -- calibrated
RS_DARK = 350.0    # s/m, night-time stomatal resistance -- calibrated
RS_R0 = 100.0      # W/m2, light-response decay scale

# ---- FAO-56 Penman-Monteith (reference-crop baseline, Allen et al. 1998) ----
# Fixed surface resistance: standard FAO-56 reference grass value.
# NOT varied or calibrated -- this is the definition of the reference crop.
RS_FAO = 70.0       # s/m, surface (bulk stomatal) resistance
U_MIN_FAO = 0.10    # m/s, leakage/fans-closed in-canopy air speed
U_MAX_FAO = 0.35    # m/s, active forced ventilation in-canopy air speed



@njit(cache=True)
def _svp_kpa(Tc):
    return 0.6108 * np.exp(17.27 * Tc / (Tc + 237.3))


@njit(cache=True)
def _setpoint(hour_local):
    if hour_local >= 6 and hour_local < 18:
        return 20.0
    elif hour_local >= 18:
        return 16.0
    else:
        return 18.0


@njit(cache=True)
def simulate(n, sub_steps, Tout, G, P_kpa, hour_local, rho_v_out,
             transpiration_mode, ach_closed, ach_open, ra, rs_min, rs_dark, rs_r0,
             rs_fao, u_min_fao, u_max_fao, trans_floor):
    """
    Forward-Euler simulation with sub-stepping (n hourly forcing rows,
    `sub_steps` sub-steps per hour -- 120 in the original report).

    transpiration_mode:
        0 = Priestley-Taylor (radiation-only, no VPD feedback)
        1 = Stanghellini (calibratable VPD-feedback via ra/rs_min/rs_dark)
        2 = FAO-56 Penman-Monteith (fixed rs=70 s/m, dynamic ra from
            in-canopy air speed interpolation: u_eff = u_min + u_fan*(u_max-u_min))

    ach_closed/ach_open/ra/rs_min/rs_dark/rs_r0/rs_fao/u_min_fao/u_max_fao passed
    explicitly (not read as module globals) so sweeps aren't silently
    ignored by numba's cached compiled constants.
    """
    dt = 3600.0 / sub_steps
    N = n * sub_steps

    Tin_h = np.empty(n)       # hourly-mean output for comparison to obs
    RHin_h = np.empty(n)
    Qtrans_h = np.empty(n)
    Qheater_h = np.empty(n)
    ufan_h = np.empty(n)

    Tin = Tout[0]
    rho_v_in = rho_v_out[0]

    for i in range(n):
        Ti_acc = 0.0
        RH_acc = 0.0
        Qtrans_acc = 0.0
        Qheater_acc = 0.0
        ufan_acc = 0.0

        To = Tout[i]
        Gi = G[i]
        Pi = P_kpa[i]
        h = hour_local[i]
        rvo = rho_v_out[i]
        sp = _setpoint(h)

        for s in range(sub_steps):
            # ---- heater: proportional controller toward setpoint ----
            err = sp - Tin
            Qheater = HEATER_KP * err if err > 0.0 else 0.0
            if Qheater > HEATER_PMAX:
                Qheater = HEATER_PMAX

            # ---- ventilation: switched ACH on setpoint+2 trigger ----
            if Tin > sp + VENT_TRIGGER_OFFSET:
                ach = ach_open
                u_fan = 1.0
            else:
                ach = ach_closed
                u_fan = 0.0
            Vdot = ach * V / 3600.0   # m3/s
            mdot_vent = RHO_AIR * Vdot
            Qvent = mdot_vent * CP_AIR * (Tin - To)

            # ---- solar, conduction ----
            Qsolar = TAU_G * Gi * AF
            Qcond = U * A_ENV * (Tin - To)

            # ---- transpiration ----
            es_in = _svp_kpa(Tin)
            Delta = 4098.0 * es_in / ((Tin + 237.3) ** 2)
            gamma = 0.000665 * Pi
            VP_in = rho_v_in * 1000.0 * (Tin + 273.15) / 2167.0   # kPa
            VPD_air = es_in - VP_in
            if VPD_air < 0.0:
                VPD_air = 0.0

            Rn = F_NET_RAD * TAU_G * Gi

            if transpiration_mode == 1:  # Stanghellini
                Rc = TAU_G * Gi
                rs = rs_min + (rs_dark - rs_min) * np.exp(-Rc / rs_r0)
                LE = (Delta * Rn + RHO_AIR * CP_AIR * VPD_air / ra) / \
                     (Delta + gamma * (1.0 + rs / ra))
            elif transpiration_mode == 2:  # FAO-56 Penman-Monteith
                # Dynamic ra derived from in-canopy air speed interpolated by u_fan
                u_eff_local = u_min_fao + u_fan * (u_max_fao - u_min_fao)
                ra_fao = 208.0 / u_eff_local
                LE = (Delta * Rn + RHO_AIR * CP_AIR * VPD_air / ra_fao) / \
                     (Delta + gamma * (1.0 + rs_fao / ra_fao))
            else:  # mode 0: Priestley-Taylor
                LE = 1.26 * (Delta / (Delta + gamma)) * Rn

            if LE < 0.0:
                LE = 0.0
            Qtrans = LE * AF

            # FIX: only apply the flat baseline floor to the old Stage-1
            # (Priestley-Taylor) term. Stage-2 (Stanghellini) already has a
            # built-in floor-like effect via the VPD-driven aerodynamic term
            # (it doesn't vanish at low light as long as VPD_air>0), so
            # stacking a second flat floor on top double-corrects and was
            # masking whether the VPD-feedback mechanism works on its own.
            if transpiration_mode == 0:
                Qtrans += trans_floor

            mdot_trans = Qtrans / LAMBDA_V

            # ---- integrate ----
            dTin = (dt / (RHO_AIR * CP_AIR * V)) * (Qsolar + Qheater - Qcond - Qvent - Qtrans)
            Tin_next = Tin + dTin

            drho = (dt / V) * (mdot_trans - Vdot * (rho_v_in - rvo))
            rho_v_next = rho_v_in + drho
            if rho_v_next < 0.0:
                rho_v_next = 0.0

            es_next = _svp_kpa(Tin_next)
            rho_v_sat = (2167.0 * es_next / (Tin_next + 273.15)) / 1000.0  # kg/m3
            if rho_v_next > rho_v_sat:
                rho_v_next = rho_v_sat

            Tin = Tin_next
            rho_v_in = rho_v_next

            Ti_acc += Tin
            VP_now = rho_v_in * 1000.0 * (Tin + 273.15) / 2167.0
            SVP_now = _svp_kpa(Tin)
            RH_now = VP_now / SVP_now * 100.0
            if RH_now > 100.0:
                RH_now = 100.0
            RH_acc += RH_now
            Qtrans_acc += Qtrans
            Qheater_acc += Qheater
            ufan_acc += u_fan

        Tin_h[i] = Ti_acc / sub_steps
        RHin_h[i] = RH_acc / sub_steps
        Qtrans_h[i] = Qtrans_acc / sub_steps
        Qheater_h[i] = Qheater_acc / sub_steps
        ufan_h[i] = ufan_acc / sub_steps

    return Tin_h, RHin_h, Qtrans_h, Qheater_h, ufan_h
