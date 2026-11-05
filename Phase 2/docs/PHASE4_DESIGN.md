# Phase 4 design: closed-loop null steering with reactive and proactive control

**Status: proposal, written before any Phase 4 code or run (2026-11-05).** No code and no results exist for the
simulator described here. The success and failure criteria in section 7 are fixed by committing this document. Changes
before any Phase 4 result exists need a new commit with a changelog entry (section 10); after the first result exists
the criteria, thresholds and the parameter table in section 8 are frozen. Open points that need a decision are in
section 9.

Inputs from earlier phases (all numbers come from saved files):

* The Stage B LSTM predicts next-slot **local-mean** interference at a victim user (dBm). Test MAE over 3 seeds:
  urban 3.0917 dB (persistence 4.9094, AR(10) 3.2314), rural 2.3278 dB (persistence 4.6413, AR(10) 2.4586)
  (`results/phase3/<scenario>/summary.json`). The prototype gate passed in both scenarios; the final gate (>= 5% better than
  the best baseline, AR(10)) failed on urban (+0.0432) and passed on rural (+0.0532).
* `AUDIT.md` C3: the Phase 3 "ZF suppression" gate was met without any null (a beam pointed at the wanted user already
  has 21-52 dB gain ratio towards the NTN direction), used one test sample, and was never applied inside the
  simulation. Phase 4 replaces it with measured interference in the simulation, always **relative to a no-null
  control run on the same channel realisation**.
* `src/phase4_metrics.py` already provides coverage probability, Shannon throughput, throughput degradation versus an
  interference-free reference, and interference reduction in linear power.

## 1. What is being tested

A terrestrial gNB with an N-element array serves its own TN user. Its leakage into nearby NTN users (victims), which the
time-series simulator models as ACIR-attenuated, shadowed and faded power, can be reduced by placing a spatial null in the
victim's direction. A null costs the gNB's own user some power and needs a decision about **when** to apply it. Three
controllers decide that (section 4). Questions:

1. How much victim interference does a geometry-based null remove, and at what cost to the gNB's own users?
2. Does deciding from an LSTM forecast (proactive) beat deciding from delayed measurements (reactive)?
3. If yes, does it beat a trivial linear forecast (AR(10)), i.e. does the neural model matter?
4. How do the answers change with array size (16/32/64), ACIR (12-45 dB) and direction error?

## 2. Geometry and the null

**Coordinates.** Scenario positions are in km (`src/entities.py`). For gNB `g` at `p_g` and a user `k` at `p_k`
(both at height 0 in the current generator), the vector `r = p_k - p_g` has length `||r||` (floor 10 m, as in the
simulator). The gNB array is a ULA whose axis is the unit vector `a_hat` (default: the x axis, fixed for all gNBs).

**Direction cosine.** `u_k = (r . a_hat) / ||r||` in [-1, 1]. The half-wavelength steering vector is
`a(u)_n = exp(j*pi*n*u) / sqrt(N)`, n = 0..N-1 (equal to `steering_vector_ula` with angle `asin(u)`).
Only `u` matters to the array: a linear array **cannot tell apart directions with the same `u`** (mirror directions about the
array axis, "cone of confusion"). A null towards a victim also nulls its mirror directions. If the gNB's own user has
`|u_s - u_v|` smaller than about `2/N`, the null removes most of the wanted signal: this is expected, it is counted in the
own-user cost, and it happens less often for larger N.

**Beams.** Un-nulled transmission uses `w0 = a(u_s)` towards the gNB's served TN user (unit norm). With nulls towards the
set `V` of protected victims (estimated directions `u_hat_v`), `A = [a(u_hat_v)]` and
`w = P w0 / ||P w0||`, `P = I - A (A^H A)^-1 A^H`. The number of simultaneous nulls is limited to `M_max = 2`
(degrees of freedom stay far above the null count for N >= 16).

**Effect on a victim `k`** (power ratio versus the un-nulled beam):

`A_k(t) = |a(u_k)^H w|^2 / |a(u_k)^H w0|^2`, bounded below by a null-depth floor `A_floor = -30 dB`
(calibration limit). `A_k = 1` when no null is applied. `A_k` can exceed 1 when a null with a pointing error lands the
victim in a lobe that was weaker before; that is a real failure mode and is not clipped.

**Ratio model (central assumption).** The Phase 3 interference series is taken as the no-null baseline: it is already
what the victim receives with the un-nulled beam. With a null the victim's leaked power is
`P_leak,null(t) = P_leak,base(t) * A_k(t)` per gNB. Consequences, deliberately accepted:

* the no-null controller reproduces the Phase 3 simulator **exactly** (a regression test, section 6), and the LSTM trained on
  Phase 3 data stays valid without retraining;
* the array size does not change the baseline interference level, only the null's selectivity and the own-user cost
  (section 3). Re-deriving the baseline from an absolute array pattern would invalidate the Phase 3 data and the LSTM
  and is out of scope.

**Cost to the gNB's own user:** `C(t) = |a(u_s)^H w|^2 / |a(u_s)^H w0|^2 = ||P a(u_s)||^2` in [0, 1] (power ratio; reported in
dB as `-10 log10 C`, 0 dB with no null).

**Direction estimate.** The gNB knows positions, not the channel. The primary configuration uses exact positions at the
slot the decision refers to. Robustness runs add a horizontal position error `e ~ N(0, sigma^2 I)` per victim and
slot (independent draws from a dedicated RNG stream), `sigma in {0, 5, 20} m`; `u_hat` is computed from the perturbed position.
A reactive null uses the geometry of slot `t - d`; victims move 1.5 m/s, i.e. 0.75 m per slot, so staleness matters
little compared with the activation delay itself.

**Served TN user.** The current generator has only NTN users, so one TN user per gNB is added from an **independent RNG
stream** (existing seeds, datasets and results do not change): uniform in azimuth, distance uniform in 50-300 m (urban) or
100-1000 m (rural) from the gNB. Fixed for the whole run (stationary).

**Multiple gNBs** (the `extended` preset): each gNB nulls independently towards its own protected victims; victim
interference is the sum of the per-gNB ratio-scaled leaks.

## 3. Antenna array model and what the 16/32/64 sweep varies

Model: uniform linear array, half-wavelength spacing at 2 GHz (lambda 0.1499 m, spacing 7.49 cm), isotropic elements,
narrowband, ideal calibration apart from the -30 dB floor, no mutual coupling, total transmit power fixed (46 dBm) with
unit-norm weights.

| N | aperture | half-power beamwidth in `u` (angle at broadside) | main-lobe null-to-null in `u` | array gain towards the served user |
|---|---|---|---|---|
| 16 | 1.12 m | 0.111 (6.4 deg) | 0.250 | 12.0 dB |
| 32 | 2.32 m | 0.055 (3.2 deg) | 0.125 | 15.1 dB |
| 64 | 4.72 m | 0.028 (1.6 deg) | 0.062 | 18.1 dB |

(Closed-form ULA values, to be confirmed by the first implementation test.)

What the sweep **varies**: angular resolution (how narrow the null is, and how close the served user and a victim can be in
`u` before the own-user cost becomes large); sensitivity of the achieved attenuation to direction error (residual
`A_k` for a given `sigma`); the own-user array gain (absolute own-user SNR, hence the throughput cost); degrees of freedom for
multiple nulls. What it does **not** vary: transmit power, the baseline interference series (ratio model), the channel, the
LSTM. Expectation (descriptive, not a success criterion): with perfect geometry the null depth is set by the -30 dB
floor for every N; with `sigma > 0`, larger N is more selective but also more sensitive to pointing error, so no monotone
trend is assumed.

## 4. Controllers

All controllers decide per slot and per victim; at most `M_max = 2` victims are nulled, chosen as the lowest estimated SINR
among those that trigger. Trigger rule (same for all): protect victim `k` in slot `t` if its estimated SINR
`S_hat - 10 log10(I_hat + N0) < gamma_trig = 15 dB` (the project's SINR target). `S_hat` is the satellite signal at the same slot
as the interference estimate (a smooth quantity), `N0` the thermal noise (-93.99 dBm).

**Sensing model (identical for all controllers, stated assumption).** Controllers observe each victim's *raw*
(un-nulled-equivalent) interference: the victim's report corrected by the gNB's own known attenuation `A_k`. This removes
the feedback that would otherwise make a reactive controller oscillate (a working null hides the interference it reacts to)
and keeps the LSTM inputs inside the training distribution. Reports are noise-free; this is optimistic for every controller
and is listed as a limitation.

| controller | `I_hat` used for the decision in slot `t` | geometry |
|---|---|---|
| **no null** | none (never nulls); `A = 1` | - |
| **reactive(d)** | raw instantaneous (faded) interference measured in slot `t - d` | positions at `t - d` |
| **proactive** | LSTM forecast of the local-mean interference for slot `t` (horizon 1, from raw inputs up to `t - 1`) | positions at `t` |

`d in {1, 2, 4}` slots (0.5, 1, 2 s); the primary configuration is `d = 1`. The LSTM is the Stage B checkpoint of the
scenario (seed 42, `results/phase3/<scenario>/final/seed_42/lstm_best.pt`) with its own normalisation; its inputs are the
three Phase 3 features built from the raw series.

**Reference controllers (additions to the three requested, not part of the success criteria):**

* *always-on null*: nulls the lowest-SINR victims every slot. Brackets the maximum reduction and the maximum own-user cost.
* *proactive-AR(10)*: same as proactive with the linear AR(10) forecast fitted on the dataset's train split. Needed to answer
  question 3, because Stage B showed the LSTM beats AR(10) by only 4-5% in MAE.

No hysteresis or hold time is used (not tuned); the switching rate and the null duty cycle are reported so chattering is visible.

## 5. Evaluation data and metrics

**Evaluation runs.** Fresh simulation runs from a seed block disjoint from every dataset seed (checked by a test), so the LSTM
has seen none of them: 10 runs per (user count in {4, 5}) x (ACIR point), per scenario, 600 slots of 0.5 s each.
Primary ACIR set: 20, 25, 30, 35, 40 dB (the LSTM's training range). Extended set: 12-45 dB every 3 dB. For ACIR outside
20-40 dB the LSTM sees out-of-distribution inputs; proactive results there are marked **OOD** and are not used for the
success criteria (retraining on the extended grid would need its own protocol amendment). Baseline preset (1 gNB, 4-5
users) is primary; the `extended` preset (3 gNBs, more users) is a secondary sensitivity.

**Primary configuration** (all success criteria refer to it): N = 64, d = 1, sigma = 0, gamma_trig = 15 dB, baseline preset,
ACIR 20-40 dB pooled with equal weight per ACIR point, urban and rural reported separately.

**Metrics** (computed over all NTN users, protected or not, all slots):

1. **Interference reduction (primary):** `100 (1 - mean(P_after) / mean(P_before))` in linear power (mW), ratio of means
   over all user-slots, with `P_before` from the no-null run of the *same* realisation. Also the mean-of-ratios variant,
   per-ACIR values, and the value restricted to slots where the controller nulled.
2. **SINR:** per-user SINR time series; report mean, median and 5th percentile (dB) and the gain over no-null.
3. **Coverage probability:** share of user-slots with SINR >= 15 dB. Reported with the **interference-free ceiling**, the
   share with SNR >= 15 dB (section 9, item 1).
4. **Throughput degradation versus the interference-free reference:** Shannon rate, B = 20 MHz, reference SINR = SNR
   (signal minus noise), `100 (1 - mean(T) / mean(T_ref))`, for no-null and each controller.
5. **Cost to the gNB's own user:** mean of `C(t)` (linear ratio) and the mean loss in dB; share of slots with loss > 3 dB; own-user
   Shannon throughput degradation versus the un-nulled beam (own-user SNR from 46 dBm + 10 log10 N - FSPL - noise). Hand
   estimate from config constants: a user at 300 m has about 64 dB SNR at N = 16 and 70 dB at N = 64, where the uncapped
   Shannon rate falls about 1.5% per dB of loss (1 dB: 1.56% / 1.43%; 3 dB: 4.69% / 4.28%). Uncapped Shannon is
   unrealistically high at these SNRs (21-23 bit/s/Hz); the dB loss is the primary cost metric and a spectral-efficiency cap
   is listed as a sensitivity, not implemented.
6. **Controller bookkeeping:** null duty cycle (share of slots with at least one null), mean nulls per slot, switching rate,
   share of slots where `A_k > 1` for a protected victim.

**Statistics.** Run-level bootstrap (resampling evaluation runs, 10,000 draws, fixed seed) for 95% confidence intervals on every
reported difference and on the primary metrics. Success criteria apply to the point estimate; the proactive-versus-reactive criterion
additionally requires the interval to exclude 0.

## 6. Implementation plan and acceptance tests (no code written yet)

Planned modules: array/geometry helpers, ratio-model closed-loop simulator on top of `simulate_time_series` (new optional
arguments, defaults unchanged), controllers, evaluation script, results under `results/phase4/<scenario>/` with `run_info`
exactly as in Stage B (clean tracked tree at run start, snapshot before running, overwrite refusal).

Acceptance tests written before the runs: the no-null controller equals the Phase 3 series bit-for-bit for the same seeds; a null with exact
geometry attenuates a victim to the floor; `C = 1 - |a_s^H a_hat_v|^2` for a single null; mirror directions share `u`; own-user cost
rises as `|u_s - u_v|` shrinks below `2/N`; `A_k` can exceed 1 with a pointing error; reactive uses measurements `d` slots
old and proactive uses horizon 1; controllers see raw (de-nulled) interference; sensing and the TN-user RNG stream do not alter existing
seeds; evaluation seeds are disjoint from dataset seeds; metric functions match `src/phase4_metrics.py`.

## 7. Success and failure, stated in advance

All thresholds apply at the primary configuration (section 5) unless stated. "Proactive" means the LSTM-driven controller.

**A. The null is useful (project-target check)**, required in both scenarios:

* A1 interference reduction >= 90% (project target) for the proactive controller;
* A2 coverage >= 95% **of the interference-free ceiling** (that is, `coverage >= 0.95 x ceiling`; the absolute 95% target is not
  attainable, section 9 item 1);
* A3 NTN-user throughput degradation versus the interference-free reference < 5% (project target);
* A4 own-user mean power loss <= 1 dB (about 1.6% uncapped Shannon throughput) and own-user throughput degradation < 5%;
* A5 no ACIR point in 20-40 dB where the proactive controller's interference reduction is negative.

**Success** = A1-A5 all met in both scenarios. **Partial** = A5 met and interference reduction >= 50% in both scenarios, with the missed
targets named. **Failure** = otherwise, in particular any of: a negative mean interference reduction (A5 violated), reduction
< 50%, or own-user loss > 3 dB. A failure is reported as such with its ACIR curve; it is not tuned away.

**B. Prediction adds value** (the project's central claim), evaluated separately and independent of A:

* B1 proactive interference reduction exceeds reactive(d = 1) by >= 2.0 percentage points in both scenarios, with the 95% interval of the
  difference excluding 0. If not met, the statement is "prediction did not help over a one-slot-delayed measurement in this model".
  (Reactive(1) acts on the last raw sample, which is a strong baseline because shadowing correlation is 0.95 per slot.)
* B2 proactive exceeds reactive(d = 4) by a larger margin than it exceeds reactive(d = 1) (a delay effect). Descriptive, not pass/fail.
* B3 the LSTM matters only if proactive-LSTM exceeds proactive-AR(10) by >= 1.0 percentage point (interval excluding 0) in a scenario. Given
  Stage B's margins over AR(10) (4.3% / 5.3% in MAE), a null result here is **expected** and will be reported as "no evidence the neural
  model adds value beyond a linear forecast".

**C. Robustness** (not part of success, always reported): sigma = 5 m must still give interference reduction >= 50% to keep "Partial";
sigma = 20 m, N = 16/32, d = 2/4, the extended ACIR grid (non-OOD controllers) and the `extended` preset are descriptive.

**What would not count as success:** reductions measured against anything other than the same-realisation no-null run; results from
OOD ACIR points used to meet A or B; thresholds, trigger level, `M_max`, floor, controllers or evaluation seeds changed after results exist.

## 8. Fixed parameters

| parameter | value |
|---|---|
| array | ULA, N in {16, 32, 64}, half-wavelength, axis = x |
| trigger level | `gamma_trig` = 15 dB estimated SINR |
| simultaneous nulls | `M_max` = 2 |
| null-depth floor | -30 dB |
| controller delays | `d` in {1, 2, 4} slots; primary 1 |
| position error | sigma in {0 (primary), 5, 20} m |
| served TN user | 1 per gNB; 50-300 m urban, 100-1000 m rural; independent RNG stream |
| bandwidth / noise | 20 MHz / -93.99 dBm (existing config) |
| evaluation | 10 fresh runs per (users x ACIR) per scenario; ACIR primary 20-40 dB; extended 12-45 dB every 3 dB |
| bootstrap | 10,000 run-level resamples, fixed seed |
| predictor | Stage B seed-42 LSTM checkpoints; AR(10) fitted on the dataset train split |
| compute | CPU only, one process; evaluation is cheap (no training in Phase 4) |

## 9. Open decisions and known limitations

1. **The 95% coverage target is unattainable under the current link budget.** Hand calculation from the config constants (to be
   confirmed by the first implementation test): satellite EIRP 80 dBm, 550 km, 2 GHz; signal -73.3 dBm at nadir (SNR 20.7 dB), -80.4 dBm at
   the pass edges (SNR 13.6 dB). SNR >= 15 dB needs a slant range <= 1063 km, i.e. `|x| <= 909 km` of the +-1125 km pass, so the
   **interference-free coverage ceiling is about 81%**, independent of any controller. Criterion A2 therefore uses the ceiling. Alternatives
   for you to decide: keep the ceiling-relative criterion (proposed); lower the coverage SINR threshold (for example to 10 dB); or
   change the satellite link budget or pass length. The last two change the channel model and would need a deliberate amendment.
2. Ratio model: the absolute array pattern is not recomputed, so the baseline interference does not depend on N.
3. Sensing is noise-free and un-nulled-equivalent; real reports would be noisy and delayed (a future sensitivity).
4. Position error is the only direction error modelled; no calibration drift, mutual coupling, wideband or elevation effects.
5. The HAP is not modelled (decision postponed); the satellite link is unaffected by the null.
6. TN user is single and stationary; own-user SINR ignores other-cell interference and any satellite interference into TN users.
7. Uncapped Shannon throughput overstates own-user rates; the dB cost is the primary cost metric.
8. The LSTM was trained on ACIR 20-40 dB; proactive results outside it are OOD and excluded from the criteria.
9. Thresholds in section 7 (>= 2 pp, >= 1 pp, 1 dB, 50%) are proposals made without having seen any Phase 4 data; please confirm or
   change them **before** implementation starts.

## 10. Changelog

* 2026-11-05: first version (proposal).
