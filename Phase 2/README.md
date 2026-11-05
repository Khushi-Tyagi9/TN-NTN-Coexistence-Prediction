# Predictive Interference Intelligence for TN–NTN Coexistence

LEO satellite (NTN) users share spectrum with a terrestrial gNB (TN). A sequence model predicts the
interference the gNB leaks into NTN users; mitigation (null steering, later PPO) uses the prediction.
See `AUDIT.md` for the review that shaped the current Phase 3 design and the list of open issues.

## Setup

Python >= 3.10 (the code uses `slots=True` dataclasses and runtime `X | Y` unions).

```bash
pip install -r requirements.txt        # simulator, training (torch, gymnasium, SB3), tests; lower-bound ranges
pip install -r requirements-lock.txt   # OR: the exact versions the suite was verified against
pip install -r requirements-dev.txt    # optional: black, flake8, isort, jupyter
python -m pytest
```

Without PyTorch/gymnasium/stable-baselines3 the suite still passes but the training and DRL tests are
**skipped** (reasons are printed, `-rs`): a green run with skips means the training code did not run.
The verified configuration is 142 passed, 0 skipped. `requirements_phase3_colab.txt` just includes
`requirements.txt`.

Training runs record `run_info` (seed, library versions, git SHA/dirty flag, command line) in the
summary JSON; `--seed` seeds the split, model initialisation, batch shuffling and PPO.

## Channel model (time-series simulator, `src/timeseries.py`)

One run = one random scenario (`src/simulation.py`: user/gNB placement from the urban or rural preset)
simulated for `n_slots` slots. Everything below is in dB/dBm per user per slot.

| Item | Model | Default |
|---|---|---|
| Carrier / bandwidth / noise figure | `RadioConfig` | 2 GHz / 20 MHz / 7 dB |
| Slot duration | `TimeSeriesConfig.slot_duration_s` | **0.5 s** |
| Run length | `n_slots` | 600 slots (300 s) |
| Wanted signal | LEO downlink, free-space path loss, fly-over along x at 7.5 km/s (slow variation), 0.3 dB jitter | 550 km altitude |
| User motion | 2-D random walk | 1.5 m/s |
| Interference path | gNB EIRP − FSPL − **ACIR** | 46 dBm gNB |
| Shadowing | AR(1) log-normal, ρ = 0.95 per slot | σ = 8 dB urban, 5 dB rural |
| Traffic load | AR(1), ρ = 0.98 | σ = 2 dB |
| Fast fading | Rayleigh power (unit mean), AR(1) on the complex coefficient | ρ from Doppler, below |
| SINR | signal / (interference + thermal noise) | |

**Fast-fading correlation from Doppler.** `f_d = v·f_c / c` and the slot-to-slot correlation of the
complex fading coefficient is `ρ = J0(2π·f_d·Δt)` (Clarke/Jakes). At 2 GHz and 1.5 m/s,
`f_d ≈ 10.0 Hz` (coherence time ≈ 42 ms), so with 0.5 s slots `ρ ≈ 0.10`: fast fading is almost
independent from one slot to the next. A float `fast_fading_correlation` overrides this (tests only).
Shorter slots raise ρ (e.g. 10 ms → ρ ≈ 0.9); changing `slot_duration_s` changes the dataset.

**Not modelled (see `AUDIT.md`):** HAP links (HAP nodes are generated but never enter SINR),
multiple gNBs, antenna patterns/gains, mitigation inside the simulation loop.

## Prediction target and baseline

* **Inputs** (window of 10 slots × 3 features): `interference_dbm` (instantaneous, *including* fast
  fading), `signal_dbm`, `sinr_db`.
* **Primary target — local-mean interference, 1 slot ahead.** The interference the next slot would
  have *without* the fast-fading term (`interference_local_dbm`: path loss, ACIR, shadowing and load
  only; equal to the expected faded power because the fading has unit mean power). Fast fading is
  unpredictable at this slot length, so it is removed from the target but kept in the inputs.
* **Secondary target — instantaneous interference** (`interference_dbm`, faded). Reported alongside;
  it contains an irreducible fading component.
* **Persistence baseline:** the last observed `interference_dbm` in the window (faded).
* **Prototype gate (gates the mitigation stage):** test MAE (dB) must be at least **15% below
  persistence MAE** (`skill = 1 − MAE_model / MAE_persistence ≥ 0.15`; fixed in advance, see
  `DEFAULT_MIN_SKILL_VS_PERSISTENCE`, CLI `--min-skill`).
* **Final gate `meets_final_target` (reported only, never blocks mitigation):** test MAE must be at
  least **5% below the best of** persistence, a 10-slot linear-power moving average and a linear
  AR(10) fitted on the train split (`DEFAULT_MIN_FINAL_SKILL = 0.05`, fixed before any training,
  CLI `--min-final-skill`). On the faded inputs persistence is a weak reference, so the prototype
  gate alone can be passed by a plain smoother; the final gate cannot.
* Also reported: MAE/RMSE in dB, the share of predictions within ±1 dB and ±3 dB, every baseline's
  MAE, both skills and both gate results, in the summary JSON and on the console, for whichever
  target the dataset in `--data-root` uses (`metadata.json`: local-mean or instantaneous). The old
  "within 10% relative error" figure is still printed but is informational only (on dBm values it is
  met by persistence).

## Datasets

```bash
python build_phase3_dataset.py --out data/phase3 --runs-per-point 8 --ablate-sinr
```

Per scenario (`urban`, `rural`) this writes `X_train.npy … y_test.npy` + `metadata.json` (channel
parameters, ρ, seeds, split, target definition) to:

| Folder | Content |
|---|---|
| `data/phase3/<scenario>/` | primary: local-mean target |
| `data/phase3/<scenario>_instantaneous/` | same runs and inputs, instantaneous target |
| `data/phase3/<scenario>_nosinr/` | local-mean target, `sinr_db` channel zeroed (ablation; shape kept at (N, 10, 3)) |

* **Split:** by simulation run, **stratified by (user count, ACIR point)**, so every operating point is
  in train, val and test and no window overlaps across splits. Counts per stratum use rounding
  (6 runs → 4/1/1). A random per-window split is refused by the loader because stride-1 windows leak
  targets into training.
* **Seeds:** run seed = `base_seed + scenario_offset + run_index`, with a per-scenario offset so urban
  and rural never share random draws. Training seeds are not yet fixed (open item in `AUDIT.md`).

## Training

```bash
python run_phase3_colab.py --data-root data/phase3/urban --scenario urban
```

See `docs/PHASE3_COLAB.md`. Results written to `results/` must come from saved result files; nothing
in this README reports model performance.
