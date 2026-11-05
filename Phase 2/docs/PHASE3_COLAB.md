# Phase 3 Colab Runbook

Install dependencies:

```bash
pip install -r requirements_phase3_colab.txt
```

Run Phase 3 (LSTM first, Transformer fallback only if LSTM misses target, then proactive ZF):

```bash
python run_phase3_colab.py \
  --data-root /content/data/phase3 \
  --scenario urban \
  --target-accuracy 0.80 \
  --antenna-count 64 \
  --zf-required-db 15.0 \
  --summary-json /content/results/phase3_summary.json
```

Enable PPO only after ZF:

```bash
python run_phase3_colab.py \
  --data-root /content/data/phase3 \
  --run-drl
```

Expected dataset layout (`.npy`):

1. Pre-split: `X_train.npy`, `y_train.npy`, `X_val.npy`, `y_val.npy`, `X_test.npy`, `y_test.npy`
2. Or unsplit: `X.npy`, `y.npy` **plus `groups.npy`** (one run id per window). The loader splits by run id
   (~70/15/15 of runs) so overlapping windows never straddle train/val/test. Without `groups.npy` loading fails;
   `--allow-window-shuffle` forces a random window split, which LEAKS test targets into training and must not
   be used for reported results.

`X` shape must be `(N, 10, 3)` and `y` must be `(N,)` or `(N,1)`.
