# greenhouse-climate-synthesis

Code for the FIT 2026 paper:

> Z. Abid, A. Muslim, M. Moetesum, N. Nashit, and Y. S. Akgül, "Generating Greenhouse Climate Data for AI-Oriented Applications via Physics-Informed Simulation: A Formulation-Sensitivity and Independent-Site Evaluation Study," *2026 International Conference on Frontiers of Information Technology (FIT)*.

The pipeline downscales hourly Open-Meteo weather to one-minute resolution and drives a coupled greenhouse energy and moisture balance to synthesize indoor temperature, RH, VPD and related variables. The `validation/` folder evaluates the model at an independent commercial greenhouse (Hoogstraten, Belgium) with three transpiration formulations.

## Layout

```
case_study/            Islamabad case-study pipeline (Section III of the paper)
  fetch_open_meteo.py      downloads hourly Open-Meteo data
  build_greenhouse_dataset.py  builds the one-minute synthetic dataset
  data/                    bundled hourly Open-Meteo input (2024-06-01 to 2026-06-01)
validation/            Independent-site evaluation (Sections IV-VI)
  model.py, driver.py      Hoogstraten model and data loading
  rev_*.py                 experiments behind the reported results
  data/                    Open-Meteo inputs for Hoogstraten 2018-2020
  revision_results/        outputs of the rev_*.py scripts (as reported)
```

## Setup

Python 3.11 was used.

```bash
pip install -r requirements.txt
```

## Hoogstraten measurements (not included)

The indoor measurements are published by Šalagovič et al. under CC BY 4.0:
Mendeley Data, DOI [10.17632/tkbkzdt5nr.2](https://doi.org/10.17632/tkbkzdt5nr.2).
Download the `Environment` folder and place it so that these files exist:

```
validation/data/hoogstraten/Environment/T&RH/2018/CENTRAL.xlsx
validation/data/hoogstraten/Environment/T&RH/2019/CENTRAL.xlsx
validation/data/hoogstraten/Environment/T&RH/2020/CENTRAL.xlsx
```

To use a data folder elsewhere, set `GREENHOUSE_DATA_DIR` to a folder containing the three `openmeteo_*.json` files and `hoogstraten/`.

## Reproducing the paper's results

Run from the `validation/` folder. All metrics use the held-out period (2018, June-September 2019, 2020); January-May 2019 is the calibration window.

| Paper item | Script | Output in `revision_results/` |
| --- | --- | --- |
| Table I (six configurations) | `rev_exp13_fair_calibration.py` | `exp13_heldout_comparison.csv`, `exp13_grid_search.csv` |
| Per-season results (Sec. V-B) | `rev_exp13_fair_calibration.py` | `exp13_heldout_per_year.csv` |
| Fig. 2 | `rev_exp24_fig2.py --compact` | `fig2_validation_compact.png` |
| Bootstrap intervals, parameter perturbation (Table I note, Sec. VI-B) | `rev_exp14_uncertainty.py` | `exp14_*.csv` |
| Day/night, monthly, hourly bias, distributions (Sec. VI) | `rev_exp15_extra_metrics.py` | `exp15_*.csv`, `exp15_hourly_bias.png` |
| Stanghellini (1987) parameter check (Sec. IV) | `rev_check_thesis_values.py` | printed |

`rev_common.py` checks that its extended simulator matches `model.simulate` exactly and prints the held-out scores of all configurations.

## Building the case-study dataset

```bash
cd case_study
python build_greenhouse_dataset.py
```

This writes `outputs/greenhouse_digital_twin_1min.csv` and `.parquet` (1,052,581 one-minute rows; about 510 MB as CSV, 200 MB as Parquet), which are not stored in the repository. The build takes about one minute. To refresh the weather input first, run `python fetch_open_meteo.py`.

The case-study dataset uses the Priestley-Taylor transpiration term (net-radiation fraction 1). The paper's independent-site results favour literature-parameterized Stanghellini for future case-study runs.

### Case-study modelling details not listed in the paper

- **Control strategy.** No control logic is specified for the case-study greenhouse, so a thermostat-style rule is assumed (`build_greenhouse_dataset.py`, CONFIG section):
  - heater: full power at or below 16 °C, off at or above 20 °C, linear in between;
  - exhaust fans: off at or below 26 °C, full power at or above 30 °C, linear in between;
  - LED lighting: on from 06:00 to 22:00 local time.
- **Solar downscaling amplitude.** The sinusoidal-physical method of Balog et al. (2023) calibrates its amplitude `A` against measured one-minute irradiance, which is not available for Islamabad. A stand-in based on the daily clearness index `kc` is used instead: `A = 0.08 (1 - kc)` on cloudy days (`kc < 0.65`) and `A = 0` on clear days (see `solar_downscale()`). The same rule is applied to direct and diffuse radiation.
- **Spin-up.** Indoor states start at outdoor conditions. The first 24 h of the case-study dataset are kept but marked with `spin_up_flag = True`; drop them for steady-state analysis. (In the independent-site evaluation the first 24 h are discarded before scoring.)

## Data sources and licences

- Open-Meteo Historical Weather API, https://open-meteo.com (CC BY 4.0).
- Hoogstraten greenhouse measurements: Šalagovič et al., Mendeley Data, DOI 10.17632/tkbkzdt5nr.2 (CC BY 4.0).
- Code in this repository: MIT licence (see `LICENSE`).
