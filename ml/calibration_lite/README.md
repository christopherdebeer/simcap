# Compact, calibration-light SIMCAP classifier

**Status: implemented and evaluated offline; NOT validated for autonomous HID.**

This experiment replaces the browser neural-network dependency with tiny classical
classifiers, tests whether a short open-hand reference can replace the existing
calibration workflow, and exports matching JavaScript and C inference code.
It deliberately leaves firmware, active models, and HID mappings unchanged.

## Results

Exact-chord accuracy means **all five finger states correct**, restricted to the
ten recorded combinations. It is not averaged finger-bit accuracy.

| Representation | User calibration | Held-out exact chord | Macro class recall | Accepted coverage | Accuracy among accepted |
|---|---|---:|---:|---:|---:|
| Raw magnetic vector | None | 86.3% | 79.7% | 86.4% | 84.4% |
| Relative magnetic vector | 30 open-hand readings | 86.3% | 79.7% | 86.4% | 84.4% |
| Soft direction | 30 open-hand readings | 75.0% | 76.1% | 53.5% | 81.7% |

These numbers use 1,162 non-saturated test samples. An additional 54 test samples
hit the numeric rail and are rejected. Relative-model coverage including those
samples is 82.6%; correctly accepted decisions account for 69.7% of all test
samples. Samples are temporally correlated, not 1,162 independent trials.

The development split selected soft direction as the best calibration-light
candidate (95.6% macro recall versus 95.0% for relative). It then fell to 76.1%
macro recall on the locked blocks. **That is a failed generalization check, not
evidence that soft direction is the preferred deployment model.** The relative
baseline is the more useful starting point mechanistically, but choosing it
after viewing this comparison is exploratory; there is no further untouched test.

Thresholds that achieved at least 95% accepted accuracy in development did NOT
maintain that reliability on held-out blocks. `accepted` means the fixed gate
passed; it does not mean the device is safe to emit a command.

## What the data can establish

Source code was inspected at `1d7927b4da11806a0286cc035b600683e3642f7e`.
Data is pinned to `287f3942b2a4d7fefb4f8ddd9b82b86e6b7ad143`.

- 43 session-named files: 41 readable JSON sessions and two Git LFS pointers.
- 44,730 samples in the readable sessions.
- Only `2025-12-31T14_06_18.270Z.json` has complete binary finger labels:
  22 labeled blocks, ten observed chords, 3,807 total samples.
- Other recordings are unlabeled or explicitly say the finger states are unknown.
  They are audited but not turned into invented ground truth or pseudo-labels.
- The labeled session's `magnet_config` says `none`, despite strong magnetic
  signatures and a magnet-type field. Metadata is not sufficient to establish
  the exact physical arrangement or whether rings were refitted.

Therefore this is a **same-session labeled-block holdout**, not cross-session,
cross-user, independently repeated-trial, or hardware-refitting validation.
Several paired blocks are adjacent parts of a single pose. Their independence
must not be overstated. There is no validated performance for the other 22 binary
chords, intermediate finger states, or everyday uncommanded activity.

## Experimental protocol

1. Read native `mx,my,mz`; apply the repository's MMC5603NJ conversion
   (`100/1024` microtesla per LSB). This assumes the same sensor/configuration as
   the recorded data; changing sensor requires an explicit manifest change.
2. Do NOT use stored calibrated/residual fields, learned Earth-field estimates,
   quaternions, or derived magnetic axes. Native axis order is preserved.
3. Trim five samples at each label boundary. Treat end indices conservatively as
   exclusive because the source formats mix conventions.
4. Reserve the first 30 samples after trimming the first open-hand block for the
   reference only. In this recording they span 1.56 seconds. They are excluded
   from training, development and test. This is explicit labeled adaptation.
5. For each chord, use its first labeled block for development: first 60% for
   training, then a five-sample embargo, then the remainder for validation.
6. Lock all subsequent labeled blocks for testing. No windows cross boundaries,
   no random sample split, no synthetic training examples, and no global
   preprocessing statistics fitted on held-out samples.
7. Compare centroid, diagonal Gaussian and shrinkage LDA for five feature
   representations. Select the algorithm per representation using development
   macro recall; ties preserve the simplest first candidate (centroid).
8. Fit a training-only class-envelope gate. Select the score-margin threshold
   using development labels only. Freeze before evaluating test blocks.

Final counts: 415 training, 224 development, 1,162 valid test, 30 reference.
Exact indices and per-file SHA256 hashes are included in `results/`.
Models are not refitted after test evaluation; the exported models are exactly
the ones whose performance is reported.

The classifier is single-sample. There is no smoothing or dwell-time benefit
hidden in these results. Some timestamps are nonmonotonic or implausibly large;
this prevents credible end-to-end latency and false-activation-per-hour claims.

## Calibration: what is removed, and what is not

Let `b` be the componentwise median of the 30 explicit open-hand readings.

**Relative model:** `x = magnetic_vector - b`.

A constant additive field offset cancels when it affects both the reference and
subsequent readings. This is exact algebra, not a learned approximation. It avoids
an ellipsoid sweep, Earth-field fit, or per-finger calibration sequence. It does
not remove changes occurring after the reference, scale changes, or individual
ring movement. It also does not make the training universal: the templates still
come from this one wearer/setup.

**Soft-direction model:** `x = (m-b) / sqrt(norm(m-b)^2 + tau^2)`, where `tau` is
three times the median reference distance from `b` (3.52 microtesla here).

This is invariant to positive common gain and constant additive offset **if the
reference, its noise, and all readings transform together**. However, removing
amplitude also removes useful pose information. Wrist rotation changes the Earth
field in device coordinates; open-hand deviations can then exceed the small
reference noise and look like a non-neutral direction. The held-out neutral-pose
recall fell to 64.7%, versus 100% for relative/raw.

Idealized stresses used offsets `[50,-30,20]` and `[200,-150,100]` microtesla and
gains 0.5/1/2. Relative predictions were unchanged by the constant offsets;
soft-direction predictions were unchanged by every tested common gain/offset.
Raw accuracy fell from 86.3% to 71.3% at the larger offset with unit gain.
**These algebraic stresses ignore clipping. They do not simulate refitting rings,
independent finger-magnet displacement, or validate physical robustness.**

Re-estimating the reference from three later open-hand blocks, excluding those
reference samples from scoring, gave relative-model accuracy of about 86.0%.
Soft direction fell to 66.2–70.8%. This is a same-session diagnostic only.

## Failures and secondary checks

- Thumb-only test recall: 4.2%; index-only: 0%. These are serious failures.
- Every thumb-only training/development sample has an anomalous timestamp. Its
  magnetic Z signature changes sharply between the first and second labeled
  blocks. This raises a data-integrity concern, but does not prove the magnetic
  readings are corrupt. No silent repair or convenient exclusion was made.
- Pinky-only has 54 rail-rejected test samples and only 28 scored test samples.
  A perfect score on those 28 must not be treated as reliable pinky tracking.
- A secondary fixed-centroid orientation diagnostic trains on raw accelerometer
  inclination >=20 degrees and tests <=-20 degrees, then reverses the split.
  Relative/raw exact accuracy is 82.5% and 84.8%; soft direction 75.9% and 78.0%.
  This is still one session, acceleration may contaminate inclination, and one
  class has only three training samples in one direction. It is not an AHRS or
  independent orientation benchmark and does not retune the exported models.
- Unknown/transition intervals were not called negative examples. False-command
  rates cannot be calculated without valid intent labels.

## Device-oriented artifacts and validation

- `runtime.cjs`: dependency-free ES5-style JavaScript core; captures a reference,
  classifies raw samples and returns a candidate/rejection. Float32 parameter
  arrays. No HID output, flashing, timers or automatic baseline adaptation.
- `generated/{raw,relative,soft_direction}.h`: float32 C99 implementations with
  no heap allocation. Supply baseline and tau from an explicit reference
  collector; baseline values are in native-axis microtesla.
- `results/model_*.json`: parameters, input contract, example reference and status.
- `export_validate.py`: compiles C on the host and checks C and Node against Python.

For each of the three exported variants, all **1,162/1,162 class decisions and
1,162/1,162 acceptance decisions matched** in both C and JavaScript. Invalid and
saturated inputs were rejected; JavaScript's 30-sample reference matched Python.
Split disjointness and the stated affine invariance are also asserted.

Each model has 130 float32 parameters: **520 bytes** of numeric model storage.
The reference needs 90 values, plus small working state. JS object overhead,
Espruino heap usage, firmware/BLE memory, runtime latency and battery life have
NOT been measured. C parity is host compilation, not a hardware test.

Example Node use (same classifier core can be incorporated into Espruino code):

```javascript
const Lite = require('./runtime.cjs');
const model = require('./results/model_relative.json');
const classifier = new Lite(model);
// Request open hand explicitly, then collect 30 actual raw sensor samples:
// classifier.addReference([mx, my, mz]);
// Later: classifier.predict([mx, my, mz]);
// Do not wire candidate predictions directly to HID keys.
```

An eventual HID integration needs an explicit arm/disarm mechanism, temporal
confirmation, release handling, re-reference controls and verified negative
examples. The current confidence gate is insufficient.

## Reproduce

From the repository root, using Python 3.11+ and Node plus a C99 compiler:

```sh
python -m pip install -r ml/calibration_lite/requirements.txt
python ml/calibration_lite/fetch_data.py --output /tmp/simcap-data
python ml/calibration_lite/run.py --data-dir /tmp/simcap-data
```

`--labeled-only` downloads just the labeled recording for faster classifier
reproduction; the resulting inventory will then cover only that recording.
No firmware upload or live hardware is needed. Results are deterministic under
the recorded dependency versions. The data fetch verifies pinned SHA256 hashes.

## Practical conclusion

The compact model and offline iteration workflow are feasible. A short neutral
reference is promising for cancelling fixed offsets without a full calibration
ritual. The available evidence does **not** validate calibration-free use,
ring-refitting robustness or unattended standalone HID. The next useful work is
to resolve recording integrity and gather one separately labeled session of a
small vocabulary; no further architecture expansion is justified by this test.
