# AUDIT — TN-NTN Coexistence Prediction (branch `phase3-timeseries`)

Scope: `src/`, `tests/`, `notebooks/`, `run_phase3_colab.py`, `build_phase3_dataset.py`, `requirements*.txt`, `docs/`, `results/`.
No code was changed. `python -m pytest` → **62 passed** (Python 3.13.3, **torch and stable-baselines3 not installed**).
All numbers below were produced by throw-away scripts run against the repo's own code (default `SimulationConfig`/`TimeSeriesConfig`, 6 runs per (users, ACIR) point, 159,300 windows per scenario, run-wise split as built by `build_phase3_dataset`). Nothing was trained; no LSTM number appears in this document.

## Verdict on the 9 suspected issues

| # | Suspicion | Verdict |
|---|-----------|---------|
| 1 | "within 10%" on dBm is trivially satisfied | **Confirmed** (C1) |
| 2 | Window-level shuffle leaks between splits | **Confirmed**, latent in the builder path, live in the unsplit-`X.npy` path (C2) |
| 3 | Saver/loader filenames never match | **Confirmed** (M1) |
| 4 | Notebook has hard-coded path + saved FileNotFoundError | **Confirmed** (m1) |
| 5 | requirements.txt lacks torch/SB3; tests don't exercise training | **Confirmed** (M2) |
| 6 | HAP not in SINR | **Confirmed** (C4) |
| 7 | Power→angle mapping unjustified; ZF outside the loop | **Confirmed, and worse than suspected** — the ZF gate passes with no nulling at all (C3) |
| 8 | Config gaps vs plan | **Confirmed** (M4) |
| 9 | Other problems | 12 further findings (M3, M5–M8, m2–m10) |

---

## CRITICAL

### C1. The 80% accuracy gate is met by "repeat the last value"
- **Where:** `src/phase3_metrics.py:61` (relative error `|pred-truth| / |truth|`), `:89`; gate at `src/phase3_prediction.py:159`, `:424`, `:485`; `src/phase3_pipeline.py:28`, `:75`.
- **Mechanism:** targets are interference in dBm (test range −127.2…−31.5 dBm urban, −130.5…−51.9 rural). A 10% tolerance is therefore 3–13 dB, and *wider the weaker the interference*. `normalized_mae` (`:22-33`) divides by mean(|dBm|) ≈ 80, so it also looks excellent for 5 dB errors.
- **Evidence (test split, no model trained):**

| Predictor | Urban within-10% | Rural within-10% | Urban within ±1 dB | Rural within ±1 dB | Urban MAE dB | Rural MAE dB |
|---|---|---|---|---|---|---|
| Persistence (last `interference_dbm`) | **78.6%** | **84.9%** | 14.1% | 15.0% | 5.11 | 4.83 |
| Linear AR(10) least-squares | **80.7%** | **88.3%** | 13.9% | 15.2% | 4.75 | 4.39 |
| Constant (train mean) | 47.8% | 56.7% | – | – | 9.92 | 8.75 |

  So a model that does nothing clever clears the ≥80% prototype gate in rural and is 1.4 pts short in urban. Any LSTM that is merely as good as a linear filter "passes". RMSE for persistence is 6.77 / 6.47 dB.
- **Fix:** add dB-domain metrics (share within ±1 dB, MAE/RMSE in dB, and skill vs persistence = `1 − MAE_model/MAE_persist`), compute the persistence baseline inside `evaluate`, make the gate "beats persistence by a stated margin" (and report ±1 dB share), keep the old metric as informational only. Add tests with a synthetic series where persistence scores >80% on the old metric but ~0% on ±1 dB.
- **Related feasibility risk (needs your decision, see "Decisions needed" below):** the series is dominated by fast fading that persistence/linear models cannot predict (±1 dB share ≈ 14–15%). A "≥80%/≥90% within ±1 dB" target is probably unreachable under this channel model; the gate should be defined relative to the predictability ceiling, not asserted.

### C2. Window-level shuffle leaks the test labels into training
- **Where:** `src/phase3_data.py:91-131` (`rng.permutation(sample_count)` at `:120`), reachable via `load_phase3_dataset` → `split_phase3_dataset` when only `X.npy`/`y.npy` exist (`:165-…`, documented as supported in `docs/PHASE3_COLAB.md`).
- **Evidence:** windows are stride-1, so window *i+1* contains window *i*'s target as its last input. After a window shuffle, **70.0%** of test targets are recoverable *exactly* (max abs error 0.0 dB) by a table lookup over training inputs only. Under the run-wise split from `phase3_dataset_builder.py`: **0.0%**. An LSTM can learn this shortcut.
- **Note:** `build_phase3_dataset` already splits by run (`src/phase3_dataset_builder.py:79-93`), so datasets from `build_phase3_dataset.py` are safe. The hole is the generic loader/splitter and the docs that advertise it.
- **Fix:** make `split_phase3_dataset` require group ids (`run_ids`) and split by group, or refuse unsplit input without them (and say so in the docs). Keep the old signature working only with an explicit `allow_window_shuffle=True` + warning. Test: leakage check (no train-input slot overlaps a test target) on the splitter output.

### C3. The ZF "15 dB suppression" gate is passed with no nulling, on a single sample, using an invented angle
- **Where:** `src/phase3_beamforming.py:132-156` (`map_interference_power_to_ntn_angle`), `:96-129` (`evaluate_zf_nulling`, suppression at `:124`); `src/phase3_pipeline.py:105-125`; `src/phase3_prediction.py:406-408` (`last_test_*` = last single test window).
- **(a) Unjustified mapping.** The angle is `nominal ± 0.12 rad` scaled linearly by where the *interference power* sits between the test-set min and max (`:154`). The simulator has no angle anywhere (`timeseries.py`, `interference.py`: no antenna pattern, no direction). Interference level does not encode direction. Both the "true" and "estimated" angle come from the same function, so the experiment is circular: it measures `|f(pred) − f(true)|`, nothing physical. The NTN user's real angle is computable from geometry (positions are in the scenario) and is not used.
- **(b) The metric is not interference suppression.** `suppression_db = desired_gain_db − ntn_gain_db` (`:124`) is a gain ratio of one beam pattern. **Baseline with NO null** (`w = a_desired`): 64 antennas → 51.7 / 32.4 / 38.6 dB at the three angles the pipeline can produce; 16 antennas → 21.1 / 21.0 / 21.9 dB. All ≥ 15 dB, i.e. the gate is satisfied by angular separation alone. Further, with 64 antennas the ZF number *increases* with pointing error (39.4 → 50.8 dB for error 0.024 → 0.24 rad) because the true angle just falls into sidelobes; it never reflects tracking quality. Perfect steering gives ~333–340 dB (numerical residual).
- **(c) One sample.** The gate is decided on `predictor.last_test_prediction` / `last_test_target` (`phase3_prediction.py:406-407`, `:467-468`), i.e. the last window of the last test run. Not an average, not even a random sample.
- **(d) Outside the loop.** ZF never touches `evaluate_scenario_sinr`/`simulate_time_series`; no SINR or interference power is recomputed with a null applied. "Interference reduction >90%" and "SINR ≥ 15 dB" cannot be derived from it.
- **(e) Naming.** `zero_forcing_weights` (`:35-…`) is a single-null projection (null-steering), not multi-user ZF (`H^H (H H^H)^-1`). Fine as a method, wrong as a label if the report says ZF precoding.
- **Fix:** (1) compute NTN direction from scenario geometry (gNB → user azimuth), have the predictor's output feed an *interference-power-aware* trigger/target rather than an angle; (2) apply the weights inside the time-stepped simulator as an extra attenuation on the leaked link toward the victim user (`link_dbm += array_gain_toward_user_db`), then report SINR/interference before vs after; (3) gate on the **change in victim interference (dB)** averaged over all test slots vs a no-null and a reactive baseline; (4) include a "no-null" control in tests so a gate that passes without nulling fails.

### C4. HAPs are generated and counted but never enter any link
- **Where:** `src/simulation.py:86-103` creates `HAPNode`s; `src/config.py:73-82`, `src/entities.py:115-146`, `reporting.py:488-492` validate/count them. `grep -i hap` on `src/interference.py` and `src/timeseries.py` → **zero hits**. `compute_user_sinr_metrics` loops only over `scenario.satellites` (`interference.py:217`) for signal and `scenario.gnbs` (`:246`) for interference; `timeseries.py:128-170` likewise.
- **Impact:** the slides/report claim joint satellite + HAP + gNB modelling. As implemented, the HAP is a placeholder that satisfies a `hap_count >= 1` validity check. SINR results are satellite-vs-gNB only.
- **Fix (pick one, your call):** (a) model HAP as a second NTN source/serving node (20 km altitude, own EIRP, FSPL + elevation-dependent loss, serving-node selection by received power; optionally HAP→user interference); or (b) remove the claim from the report/slides and document the HAP as out-of-scope. Add a test that changing HAP altitude/power changes SINR.

---

## MAJOR

### M1. `save_phase3_dataset_split(prefix=…)` writes names `load_phase3_dataset` never reads
- **Where:** `src/phase3_data.py:214-221` writes `<prefix>_X_train.npy`; `:135-149` looks for `X_<scenario>_train.npy`.
- **Evidence:** saved with `prefix="urban"` → `urban_X_test.npy, urban_X_train.npy, …`; `load_phase3_dataset(d, scenario="urban")` → `FileNotFoundError: No compatible Phase 3 dataset files found`. The builder dodges this by not passing a prefix (`phase3_dataset_builder.py:150`), and no test round-trips a prefixed save.
- **Fix:** make the saver emit `X_<scenario>_<split>.npy` (rename the parameter to `scenario`, keep `prefix` as a deprecated alias mapping to the same names), and have the loader accept both legacy names with a warning. Round-trip test for no-prefix and prefixed layouts.

### M2. The training stack is effectively untested and not installable from `requirements.txt`
- **Where:** `requirements.txt` (no torch / gymnasium / stable-baselines3; they exist only in `requirements_phase3_colab.txt`).
- **Evidence:** torch is not installed here and 62 tests pass. Coverage: `phase3_prediction.py` **30%**, `phase3_drl.py` **23%**, `phase3_metrics.py` **37%**, `phase3_beamforming.py` **62%**. `tests/test_phase3_prediction.py` monkeypatches `train_lstm_forecaster`/`train_transformer_forecaster`, so `_fit_model`, `_predict`, the models, normalization, checkpoints, and the PPO env never execute. `run_phase3_colab.py` and `build_phase3_dataset.py` have no tests. A shape bug in the LSTM would not be caught.
- **Fix:** add torch/gymnasium/SB3 to a single `requirements.txt` (or `[ml]` extra) with version pins; add tiny-model CPU tests (2 epochs, 64 samples) guarded by `pytest.importorskip("torch")` that train, predict, checkpoint, and reload; add metrics tests; add a CI note so a skipped suite is visible (not silently green).

### M3. Nothing in the pipeline compares against a baseline, and LSTM training is not reproducible
- **Where:** `src/phase3_prediction.py` (no `torch.manual_seed`, `np.random.seed`; `grep manual_seed` → 0 hits; `DataLoader(shuffle=True)` at `:172` unseeded). The CLI `--seed` only feeds the dataset split.
- **Fix:** seed torch/numpy/CUDA and the DataLoader generator from the config seed; record seed, versions and git SHA in the saved summary JSON.

### M4. Config and metrics do not cover the plan's targets
- **ACIR:** `src/config.py:25-27` → 20–40 dB, 5 points; plan is 12–45 dB.
- **Antennas:** no antenna field anywhere in `config.py`; `64` is only a default argument in `phase3_pipeline.py` / `run_phase3_colab.py`. Plan: 16–64.
- **Counts:** `BASELINE_GNB_COUNT = 1` (`config.py:7,97`; `entities.py:258-259`), users forced to 4–5 (`config.py:241`, `entities.py:260`, `simulation.py:48`). 1 gNB means no aggregate TN interference, no inter-gNB coordination.
- **Missing metrics:** `grep` for coverage-probability, throughput, interference-reduction, reactive, proactive in `src/` (excluding Phase 3 beamforming/pipeline) → **no hits**. Baseline SINR is a mean/min/max only; there is no outage, rate, or before/after comparison.
- **Fix:** widen `ACIRSweepConfig` to 12–45 (explicit grid, e.g. 12, 15, …, 45); add `antenna_counts=(16,32,64)` to config; add `coverage_probability(sinr, thr)`, `throughput_bps = B·log2(1+SINR)` (with degradation vs a no-interference reference), `interference_reduction_pct`; relax the 4–5 user / 1 gNB baseline check behind an explicit flag (keep the baseline preset, add an "extended" preset).

### M5. Prediction target is physically capped and the fast-fading model is not physical
- **Where:** `src/timeseries.py:45` `fast_fading_correlation = 0.7` at `slot_duration_s = 0.5` (`:38`) and `user_speed_m_s = 1.5` (`:40`).
- **Evidence:** at 2 GHz and 1.5 m/s, Doppler ≈ 10 Hz → coherence time ≈ 42 ms, i.e. a 500 ms slot is ~12 coherence times; Rayleigh samples one slot apart should be essentially uncorrelated (ρ≈0), not 0.7. The 0.7 choice makes fast fading look partially predictable and also hides that the unpredictable component sets a hard error floor (persistence/linear ±1 dB share ≈ 14–15% already). The same module also shares `seed` between scenario and series RNGs.
- **Fix:** either shorten the slot (≈ 5–20 ms) or set ρ from the Doppler (`J0(2π f_d Δt)`), and decide explicitly whether the model predicts the *fading-averaged* interference (smoothed target) or instantaneous interference. Document the choice in the README.

### M6. PPO is a toy, detached from the simulator, and its trained model is discarded
- **Where:** `src/phase3_drl.py:101-177` env; `src/phase3_pipeline.py:126-134`.
- **Evidence:** the "predicted interference" is `base_interference_db (−95) + N(0,1)` resampled every step (`:126-128, :150, :165`), independent of the actions, the predictor, ACIR, or the ZF; reward is the difference of two noisy SINRs (`:169`); the PPO object returned by `train_ppo_mitigator` is not stored (`phase3_pipeline.py:128`), `ppo_trained=True` is just a flag. The only ZF link is the finite check in `assert_zf_validated` (`:91-98`), which would `raise ValueError` if suppression were `inf`.
- **Fix (Stage C/D):** build the env on `simulate_time_series` (state = predicted interference, SINR history; actions = power / null depth / beam selection), reward = SINR or throughput with a per-step baseline, save and evaluate the policy against reactive and proactive-ZF controllers. Do not start until ZF is fixed (C3).

### M7. Saved baseline results disagree with the current code
- **Evidence:** all 60 rows of `results/baseline/run1/metrics/baseline_run_summary.csv` carry `"gnb_count": 3`, but the current code enforces exactly 1 gNB (`config.py:97`, `entities.py:258`) and `reporting.py:497` rejects any other value. These results were produced by a different version and cannot be regenerated; they also use 3 runs/point vs `monte_carlo_runs = 20` (`config.py:185`). Any figure in the report built from this folder is not reproducible.
- **Fix:** regenerate with current code under a documented command; store config + git SHA beside results; treat the existing folder as archived/unused.

### M8. Documentation does not describe the project
- `ENVIRONMENT_VISUAL.md:26-45,154`: lists `channel_models.py`, `system_model.py`, `sinr_engine.py`, `utils.py` (none exist; real modules are `channels.py`, `entities.py`, `interference.py`, …), 3.5 GHz (code: 2 GHz, `config.py` RadioConfig), 40 dBm gNB / 16–64 antennas (code: 46 dBm, no antennas), ACIR 12–45 (code: 20–40), Urban 10/km² · Rural 1/km² (code: 3.5 / 0.8), LEO 600 km (code: 550 km), 3000 MC samples (code: 20 runs).
- `docs/PHASE3_COLAB.md`: example uses `--data-root /content/data/phase3 --scenario urban` but `build_phase3_dataset.py` writes `data/phase3/urban/X_train.npy`, so the example path is wrong by one directory; describes unsplit `X.npy` as fine (see C2); never mentions the `X_<scenario>_<split>` layout the loader supports.
- No `README.md`; no statement of the channel model, the python version (code needs ≥3.10: `slots=True`, `X | Y` at runtime, `entities.py:184`), or how to reproduce results.
- `pytest.ini:21-26` declares markers `channel`, `system`, `sinr` for modules that do not exist and no test uses them (`grep pytest.mark` → only `parametrize`).

---

## MINOR

- **m1. Notebook** `notebooks/phase3_colab_training.ipynb` cell 3: `PROJECT_ROOT = Path('c:/Users/rockb/OneDrive/Desktop/Projects/BTP/Phase 2')` (hard-coded, another user's machine); the committed output of that cell is `FileNotFoundError: run_phase3_colab.py not found …` (executed on Colab). Cell 3 local branch points `DATA_ROOT` to `results/baseline/data`, which nothing creates. Cell 7 prints only the old within-10% metric. **Fix:** derive root from `Path.cwd()` / env var, point at `data/phase3/<scenario>`, clear outputs (`nbstripout`), print the new dB metrics.
- **m2. Dead code:** `estimate_ntn_angle_from_channel` and `estimate_ntn_angle_from_sin` (`phase3_beamforming.py:159,192`) have 0 callers and 0 tests; `BaseChannelModel`, `ChannelSample`, `PathLossBreakdown`, `FreeSpaceChannelModel` are used only inside `channels.py` + one test; `SatelliteConfig.rx_gain_dbi`/`SatelliteNode.rx_gain_dbi` is never used in a link budget (no UE or gNB antenna gain either); `HAPConfig.drift_radius_m`/`anchor_xy_km` only sample a static position (and no HAP motion occurs in the time series); `Phase3DatasetSplit.sample_count` is test-only.
- **m3. Unused dependencies:** `scipy`, `pandas` are in both requirement files but never imported in `src/` or the CLI scripts (scipy was used only in my audit script); `black/flake8/isort` listed but no config.
- **m4. Unstratified, float-fragile run split:** `phase3_dataset_builder.py:82-85`: `n_val = floor(n_runs * (1 − 0.70 − 0.15))`; `1 − 0.70 − 0.15 = 0.15000000000000002` happens to work for 60 runs but is fragile for other counts. Splits are random over (users, ACIR, run) so a split can miss an ACIR level entirely. Also val has 24,190 and test 23,600 windows only because of user-count mix. **Fix:** stratify by (users, ACIR), compute counts with `round`.
- **m5. Seed reuse across scenarios:** `DatasetBuildConfig.base_seed = 1000` (`phase3_dataset_builder.py:42`) gives urban and rural identical per-run seeds, so their random draws are correlated (different spread/density only rescales positions). Offset the seed by scenario.
- **m6. Redundant feature:** `sinr_db` is a deterministic function of the other two features (`timeseries.py` end), so the LSTM input has rank 2, and `y` is a function of one input channel; a leakage-safe ablation (drop `sinr_db`, add user/gNB geometry or ACIR as static input) should be tried.
- **m7. Gym API:** `InterferenceMitigationEnv.reset` (`phase3_drl.py:144`) does not call `super().reset(seed=seed)`; `action_space = Discrete(4)` hard-coded.
- **m8. Test realism:** `tests/test_phase3_pipeline.py` and `test_phase3_prediction.py` use interference "powers" of 0.5–1.5 (linear units) while the real target is dBm; `test_phase3_pipeline.py:98-114` asserts `zf_gate_passed` which, per C3, would also pass with no null. These tests should be kept (per your rule), and new ones added next to them.
- **m9. Repository hygiene:** the Phase 3 time-series files are untracked (`build_phase3_dataset.py`, `src/timeseries.py`, `src/phase3_dataset_builder.py`, two test files) on a branch with one commit; `Phase 2.zip` is gitignored (OK) but the `.docx` report/guide are committed as binaries.
- **m10. Metrics naming:** `PredictionMetrics.within_10pct_accuracy` hard-codes the tolerance in the field name; the new dB metrics need a versioned schema so old `phase3_summary.json` files remain readable.

---

## Decisions needed before Step 2

1. **Accuracy target definition.** With the current channel model, persistence and a linear filter reach only ≈14–15% within ±1 dB (MAE ≈ 4.4–5.1 dB). I propose to gate on *(a)* LSTM MAE (dB) at least X% lower than persistence MAE and *(b)* report ±1 dB / ±3 dB shares, rather than promise 80/90% within ±1 dB. What improvement margin X do you want (I suggest 15%)? Alternatively we make the channel more predictable (M5) — but that must be justified physically, not tuned to hit a number.
2. **HAP (C4):** implement HAP as a second serving/interfering NTN node, or drop the claim from the report?
3. **ZF gate (C3):** OK to replace the angle-from-power mapping with geometry-derived angle + in-simulator application, and to redefine the gate as measured victim-interference reduction vs a no-null control?

## Proposed Step-2 order (one commit each, tests added with every commit)
1. C1 dB metrics + persistence baseline + new gate (keep old metric, informational).
2. C2 group-aware split (+ leakage test) and M1 saver/loader naming round-trip.
3. M2 torch-guarded training tests + requirements consolidation; M3 seeding.
4. C3 geometry-based angle, in-loop ZF with no-null control; C4 HAP (per your decision).
5. M4 config (ACIR 12–45, antenna sweep, new metrics) — needed for Stage D.
6. M5 fading model, M7 regenerate baseline, M8/m1 docs + notebook, minors.
