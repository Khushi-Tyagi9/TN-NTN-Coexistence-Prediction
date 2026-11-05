# Preliminary LSTM runs (exploratory; validation split only)

**These are exploratory results, not the final protocol.** They come from `run_preliminary.py` and are scored on the
**validation** split only. The test split was not read (only the four train/validation `.npy` files are opened) and
**no gate is evaluated**. Validation was also used for early stopping and best-epoch choice, so the numbers are
mildly optimistic. One seed, no hyperparameter search. Every number below is copied programmatically from the
`summary.json` files in `urban/`, `rural/` and `urban_instantaneous/`.

Config (all three runs): hidden 96, 1 layer, lr 0.001, batch 64,
seed 42, at most 40 epochs, early-stopping patience 6 on validation loss.
Target for `urban` and `rural`: next-slot local-mean interference (`interference_local_dbm`, fast fading removed from the target, kept in the
inputs). Target for `urban_instantaneous`: the faded next-slot interference itself (`interference_dbm`), same inputs.

## Validation results

| scenario | LSTM MAE (dB) | RMSE (dB) | within 1 dB | within 3 dB | persistence MAE | 10-slot avg MAE | AR(10) MAE | best baseline | skill vs persistence | skill vs best baseline | best epoch / epochs run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| urban | 3.097 | 3.891 | 0.205 | 0.563 | 4.933 | 3.906 | 3.227 | linear_ar_10 | +0.372 | +0.040 | 18 / 24 |
| rural | 2.325 | 2.921 | 0.269 | 0.700 | 4.607 | 2.654 | 2.472 | linear_ar_10 | +0.495 | +0.060 | 2 / 8 |
| urban_instantaneous | 5.285 | 6.779 | 0.122 | 0.357 | 6.350 | 6.209 | 5.375 | linear_ar_10 | +0.168 | +0.017 | 3 / 9 |

Skill = 1 - MAE_LSTM / MAE_baseline. The best baseline is linear AR(10) (least squares, fitted on the train split) in all three
runs. Read with the caveats above: one seed, validation also used for early stopping, no gate evaluated.

Plots per scenario: `training_curves.png`, `actual_vs_predicted_val.png` (validation, 200-slot window).

## Provenance (both SHAs, dirty-tree note)

| scenario | git SHA recorded | tree dirty | tracked files differing from that commit | torch threads | command line | dataset generated at |
|---|---|---|---|---|---|---|
| urban | `d0a58a0` | True | src/config.py, src/phase3_search.py, tests/test_stage_b.py, run_stage_b.py | 12 | `--scenario urban --threads 12` | 0c972a0 |
| rural | `1911e83` | True | src/phase3_search.py, tests/test_stage_b.py, run_stage_b.py | 12 | `--scenario rural --threads 12 --eta-limit-minutes 30` | 0c972a0 |
| urban_instantaneous | `1911e83` | True | run_stage_b.py, src/phase3_search.py, tests/test_stage_b.py, results/figures/baseline_errors.json | 8 | `--scenario urban --variant instantaneous --threads 8` | 0c972a0 |

* **Two different SHAs.** The urban run recorded `d0a58a0`; the rural and the instantaneous-target runs recorded `1911e83`. Between them three commits were made
  (`774431d` Phase 4 config, `5b07dc4` Phase 4 metrics, `1911e83` run-info snapshot).
* **Evidence that both runs used identical training/data code.** Command and result:

  ```
  git diff d0a58a0 1911e83 --stat -- src/phase3_*.py src/timeseries.py src/phase3_dataset_builder.py src/reproducibility.py run_phase3_colab.py
  (empty output: none of those files changed between d0a58a0 and 1911e83)
  ```

  The only other files that differ between the two commits are `run_preliminary.py` (provenance snapshot moved before training),
  `src/config.py` (additive Phase 4 options), `src/phase4_metrics.py` (new) and three test files. `run_preliminary.py` imports only
  `src.phase3_data`, `src.phase3_metrics`, `src.phase3_plots`, `src.phase3_prediction`, `src.phase3_preliminary` and
  `src.reproducibility`; `src.config` and `src.phase4_metrics` are not imported by it.
* **Dirty tree.** In all three runs `git_dirty` is true. `src/phase3_search.py`, `tests/test_stage_b.py` and `run_stage_b.py` carry the
  uncommitted Stage B time-budget amendment (grid and budget decisions postponed); the preliminary run does not import them.
  The urban run's list additionally shows `src/config.py`: its provenance snapshot was taken at the **end** of the run, and
  by then it held uncommitted Phase 4 preparation, written while the run was training. Those changes were additive (defaults unchanged,
  golden-tested against values captured before the change) and the run never imported that module. The rural run took its snapshot
  **before** training, after those changes were committed. The instantaneous-target run (same SHA, snapshot before training) additionally lists
  `results/figures/baseline_errors.json`: the figures script had just been re-run, changing only that file's provenance fields (timestamp, SHA,
  dirty list; the numbers are identical).
* **Thread count differs.** The urban and rural runs used 12 torch threads; the instantaneous-target run used 8 (the requested 6-8 range).
  Thread count can change floating-point summation order, so exact reproduction needs the same count.
* **Known cosmetic bug in the dirty-file list:** entries show stray quote characters (git quotes paths containing a space and the
  parser strips one character). The content is readable. It is not fixed yet because `src/reproducibility.py` is part of the
  code the two runs share; fix it in a later commit.

## Training time: slowed epochs

| scenario | epochs | min s/epoch | median s/epoch | max s/epoch | wall clock (s) |
|---|---|---|---|---|---|
| urban | 24 | 10.5 | 36.9 | 50.9 | 785 |
| rural | 8 | 16.4 | 17.8 | 43.7 | 189 |
| urban_instantaneous | 9 | 15.6 | 34.5 | 36.6 | 259 |

Seconds per epoch (in order):
* urban: 14, 11, 12, 10, 12, 38, 36, 40, 35, 34, 43, 42, 44, 39, 30, 40, 44, 31, 19, 39, 43, 43, 33, 51
* rural: 22, 44, 36, 18, 17, 17, 16, 18
* urban_instantaneous (8 threads): 22, 18, 34, 36, 36, 37, 35, 23, 16

* Urban epochs 1-5 took 10-14 s; from epoch 6 to the end they took 19-51 s (median over all epochs above). The **cause is not
  established**. The slowdown began about a minute into the run. My concurrent test runs came much later (about nine minutes in), so
  they do not explain the onset, although they may have added to it. Circumstances, not a diagnosis: a 13th-gen i5-1340P laptop
  (hybrid performance/efficiency cores), 12 torch threads, an observed clock of 1504 of 1900 MHz, and several browser and desktop
  processes open.
* Rural epoch times were 16-44 s. **Timing numbers from these runs are not representative of the hardware's best speed** and must
  not be used to plan the final protocol without re-measuring.
* A first rural attempt was stopped by the script's own ETA guard (console output only, not saved: epoch 1 took 35.0 s, ETA 23.3 min
  against the 20 min limit) and wrote nothing. Rural was restarted with `--eta-limit-minutes 30`, which is recorded in its `argv`.

## Urban: instantaneous vs local-mean target (same config, validation split)

| metric | local-mean target | instantaneous target |
|---|---|---|
| LSTM MAE (dB) | 3.097 | 5.285 |
| persistence MAE (dB) | 4.933 | 6.350 |
| 10-slot avg MAE (dB) | 3.906 | 6.209 |
| AR(10) MAE (dB) | 3.227 | 5.375 |
| within 1 dB | 0.205 | 0.122 |
| within 3 dB | 0.563 | 0.357 |
| skill vs persistence | +0.372 | +0.168 |
| skill vs best baseline | +0.040 | +0.017 |

Facts from the table: every predictor has a larger error on the instantaneous target, and the LSTM's margin over the best baseline is smaller.
This is consistent with an irreducible fast-fading component in the instantaneous target (interpretation; one seed, validation only).

## Not done

Test split, gates, hyperparameter search, 3-seed runs.
