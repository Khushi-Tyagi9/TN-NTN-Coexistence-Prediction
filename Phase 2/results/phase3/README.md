# Stage B results: LSTM, local-mean target (final protocol)

Protocol: `docs/STAGE_B_PROTOCOL.md`. The test split was evaluated **once per config/seed**; both gates are evaluated on the
**mean test MAE over seeds 42, 43, 44** per scenario. The gate thresholds (prototype: skill vs persistence >= 0.15;
final: skill >= 0.05 vs the best of persistence, 10-slot linear-power average and AR(10) fitted on train), the grid
and the channel model were not changed after seeing results. Every number below is generated from the saved JSON files in
`urban/` and `rural/` (see `run_info` in each).

**The hyperparameter search was reduced because of a time limit:** urban only, 5 configs, 40 epochs / patience 6,
instead of the originally declared grid (hidden {64,96,128} x lr {1e-3,3e-4} x layers {1,2} = 12 configs, 120 epochs, patience 12). The urban-selected config is reused unchanged for
rural; there was no rural search.

## Gate results (mean test MAE over 3 seeds)

| scenario | LSTM mean MAE (dB) | persistence | 10-slot avg | AR(10) | skill vs persistence | prototype gate (>= 0.15) | skill vs best baseline (AR(10)) | final gate (>= 0.05) |
|---|---|---|---|---|---|---|---|---|
| urban | 3.0917 | 4.9094 | 3.9770 | 3.2314 | +0.3702 | PASS | +0.0432 | **FAIL** |
| rural | 2.3278 | 4.6413 | 2.6713 | 2.4586 | +0.4985 | PASS | +0.0532 | PASS |

* **Urban: prototype gate passed, final gate FAILED** (skill +0.0432, -0.0068 relative to the 0.05 threshold; no single seed reaches it).
* **Rural: both gates passed**, the final gate with a margin of only +0.0032 above the threshold
  (per-seed skills +0.0551, +0.0537, +0.0508).
* The Transformer was not required (prototype gate passed in both scenarios).

## Search (urban, seed 42, validation only; selected by lowest validation loss)

| # | hidden | layers | lr | val loss | val MAE (dB) | best / run epochs | wall (s) |
|---|---|---|---|---|---|---|---|
| 1 | 96 | 1 | 0.001 | 0.11267 | 3.0965 | 18 / 24 | 464 |
| 2 | 64 | 1 | 0.001 | 0.11260 | 3.0922 | 21 / 27 | 167 |
| 3 | 64 | 1 | 0.0003 | 0.11237 | 3.0906 | 26 / 32 | 312 |
| 4 | 128 | 1 | 0.001 | 0.11263 | 3.0934 | 15 / 21 | 247 |
| 5 **(selected)** | 128 | 1 | 0.0003 | 0.11231 | 3.0903 | 26 / 32 | 334 |

Selected: hidden 128, 1 layer, lr 0.0003, at most 40 epochs, patience 6.
The five trials differ by about 0.0004 in validation loss, so the choice among them is nearly arbitrary.

## Urban: test split, per seed

| seed | test MAE (dB) | RMSE (dB) | within 1 dB | within 3 dB | skill vs persistence | skill vs best baseline | best / run epochs |
|---|---|---|---|---|---|---|---|
| 42 | 3.0881 | 3.8914 | 0.2072 | 0.5643 | +0.3710 | +0.0443 | 26 / 32 |
| 43 | 3.0946 | 3.9009 | 0.2066 | 0.5645 | +0.3697 | +0.0423 | 6 / 12 |
| 44 | 3.0924 | 3.8960 | 0.2069 | 0.5644 | +0.3701 | +0.0430 | 10 / 16 |
| **mean** | **3.0917** | 3.8961 | 0.2069 | 0.5644 | | | |
| **std** (sample, n=3) | 0.0033 | 0.0048 | 0.0003 | 0.0001 | | | |

## Rural: test split, per seed (same config as urban)

| seed | test MAE (dB) | RMSE (dB) | within 1 dB | within 3 dB | skill vs persistence | skill vs best baseline | best / run epochs |
|---|---|---|---|---|---|---|---|
| 42 | 2.3231 | 2.9330 | 0.2744 | 0.6969 | +0.4995 | +0.0551 | 20 / 26 |
| 43 | 2.3265 | 2.9346 | 0.2716 | 0.6973 | +0.4987 | +0.0537 | 8 / 14 |
| 44 | 2.3337 | 2.9440 | 0.2729 | 0.6957 | +0.4972 | +0.0508 | 10 / 16 |
| **mean** | **2.3278** | 2.9372 | 0.2730 | 0.6966 | | | |
| **std** (sample, n=3) | 0.0054 | 0.0059 | 0.0014 | 0.0008 | | | |

Plots per scenario: `training_curves.png` (all seeds), `actual_vs_predicted.png` (seed 42, test split: scatter, 200-slot window, error CDF).
Per-seed folders `final/seed_<s>/` hold the summary, checkpoint and test predictions.

## Reading these results (caveats)

* The seeds change model initialisation and batch shuffling only; **each scenario has a single simulated dataset** (all runs of a scenario share one
  data hash, see Provenance), so the std measures training variance, not data variance. The final-gate outcomes (urban fail, rural pass) rest on one data realisation each.
* The LSTM's margin over the best baseline is 4.3% (urban) and 5.3% (rural) in MAE: it is close to what a linear AR(10) filter achieves on this target.
* The config was chosen on urban validation and applied to rural unchanged.
* Only the local-mean target is part of Stage B. The instantaneous-target and sinr-ablation datasets exist but were not run here
  (the instantaneous-target urban run is in `preliminary/`, validation only).

## Provenance

| run | git SHA | tree dirty | dirty files | torch threads | X_test.npy sha256 (first 12) |
|---|---|---|---|---|---|
| search (urban) | `2dfe9ff` | False | none | 8 | `c3bb1fd55cc0` |
| urban seeds 42/43/44 | `9d4b3dc` | False | none | 8 | `c3bb1fd55cc0` |
| rural seeds 42/43/44 | `e7734c6` | False | none | 8 | `e7a8c3858a34` |

All runs started from a clean tracked tree. Code changes between the search commit and the rural run (everything except `results/`):

```
git diff 2dfe9ff e7734c6 --stat -- . ':(exclude)results'
(empty: no code changed; only result files were added between these commits)
```

Library versions, seeds and command lines are in each `summary.json` under `run_info`. Earlier exploratory validation-only runs are in `preliminary/`.
