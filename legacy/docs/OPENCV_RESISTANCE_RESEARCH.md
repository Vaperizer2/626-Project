# OpenCV resistance-estimation feasibility (draft)

**Date:** 2026-08-13  
**Scope:** feasibility and the smallest credible secondary component for this
repository's planned OpenCV demo. This is a research/design note only; no
implementation is included.

## Decision in one paragraph

Resistance estimation is feasible as a constrained hybrid pipeline for a
single, visible, axial through-hole resistor with readable colour bands. The
smallest useful first version is not a second learned regressor: use the
existing component classifier only as a coarse "resistor" gate, then use
OpenCV to segment/rectify the resistor body, measure colour in band-shaped
regions, decode the band sequence against IEC 60062/manufacturer tables, and
return a nominal value plus tolerance and an explicit abstention state. A
learned secondary model becomes justified only after a representative,
component-identity-separated dataset shows that the classical decoder fails
on localization, band boundaries, or colour ambiguity. The current checkout
does not contain the OpenCV demo, the necessary annotations, or a resistance
checkpoint, so it cannot train or validate that secondary model today.

## Inspected repository facts

These are observations from the current checkout, separate from the design
recommendations below.

| Area | Observed fact | Status for resistance work |
|---|---|---|
| Project shape | The repository is a notebook-based 20-class electronic-component image-classification project. The README explicitly leaves "the later OpenCV/value-recognition pipeline" as future work. ([README.md](README.md), [notebooks/01_primary_comparative.ipynb](notebooks/01_primary_comparative.ipynb)) | No OpenCV demo is present. |
| OpenCV implementation | Repository search found no cv2 import, VideoCapture, OpenCV threshold/contour/segmentation code, camera calibration code, or OpenCV DNN/detector integration. pyproject.toml declares no OpenCV package. ([pyproject.toml](pyproject.toml)) | Camera, detection, segmentation, and calibration components are absent. |
| Current labels | The frozen split has these labels: battery holder, breadboard, capacitor, connector, crystal oscillator, diode, fuse, heat sink, ic chip, inductor coil, jumper wire, led, microcontroller board, potentiometer, power supply module, relay, resistor, switch, transformer, transistor. ([data/splits/split_meta.json](data/splits/split_meta.json)) | resistor is one coarse image-level class; there are no band, value, tolerance, body-box, or mask labels. |
| Split size | The frozen split records 2,438 train, 522 validation, and 523 test images; resistor contributes 117/25/25 images. ([data/splits/split_meta.json](data/splits/split_meta.json)) | Only 167 resistor samples are in the current split, and none has resistance ground truth. |
| Resistor-image scope | A read-only spot check found heterogeneous images: an axial banded resistor in a scene, a power/package resistor with printed text, and a resistor-type collage/diagram. Examples are under [data/raw/electronic-component-recognizer/resistor/resistor/](data/raw/electronic-component-recognizer/resistor/resistor/). The README also warns that the web-scraped dataset contains noise and mislabeled/unrelated images. ([README.md](README.md)) | The current class is not an axial-colour-band dataset; scope must be narrowed or annotated before value decoding. |
| Model outputs | The notebook builds a ComponentDataset from image paths and class names, then trains image classifiers. It has no object-detection, segmentation, band-sequence, or scalar-resistance head. ([notebooks/01_primary_comparative.ipynb](notebooks/01_primary_comparative.ipynb)) | Existing models can gate "resistor"; they do not localize a body or estimate Ω. |
| Checkpoints | The local ignored directory notebooks/outputs/weights/ contains classification artifacts including primary_resnet18.pt, primary_small_cnn.pt, resnet18.pt, resnet50.pt, efficientnet_b0.pt, scratch-CNN variants, and Optuna/playground variants. Metadata inspected in the .pt files declares the same 20 class names and no resistance output. The repository-root outputs/weights/ is empty. ([HANDOVER.md](HANDOVER.md), [.gitignore](.gitignore)) | Resistance/detector/segmenter checkpoints are absent. The existing ResNet checkpoint must not be treated as a resistance model. |
| Resistance annotations | No resistance-value/tolerance table, band annotation file, body boxes/masks, camera calibration file, or resistance model artifact was found in the checkout. | A learned secondary model cannot be trained from the present assets. |

The HANDOVER.md describes additional artifacts from a remote run, but those
files are not present under this checkout's root outputs/weights/; this note
reports the local filesystem, not the remote inventory.

## What the primary sources establish

### Detection, segmentation, and rectification

OpenCV's inRange tutorial demonstrates converting a BGR frame to HSV and
thresholding a lower/upper HSV range into a binary mask. OpenCV's contour
tutorial says contours are found in a binary image and recommends thresholding
or edge detection first. The shape API provides connected-component labels and
statistics, findContours, and minAreaRect, which returns a rotated minimum
area rectangle. The color-conversion API exposes BGR/RGB-to-HSV and BGR/RGB-to-
Lab conversions. [O1][O2][O3][O4]

That supports a small classical detector for a controlled scene:

1. Convert the frame consistently from OpenCV's BGR convention to HSV and/or
   Lab.
2. Build a foreground/body mask, clean it with conservative morphology, and
   use connected-component statistics plus contour geometry to reject small,
   fragmented, or non-elongated candidates.
3. Use the candidate's rotated rectangle to estimate the resistor axis and
   body crop. getRotationMatrix2D/warpAffine are the appropriate OpenCV
   primitives for deskewing a crop. [O3][O4][O7]
4. Exclude leads and end margins before sampling bands. A contour rectangle is
   a geometric proposal, not proof that the object is a resistor; failed or
   ambiguous geometry must abstain.

For a single resistor on a neutral background this is a reasonable baseline.
For clutter, multiple components, severe occlusion, or arbitrary resistor
types, a learned detector/segmenter may eventually be needed. The current
repository has neither the boxes/masks nor the demo code to support that
option.

### Colour-band decoding

IEC 60062:2016 is the relevant international marking-code standard; its
official description explicitly covers colour assignment, resistance-value
and tolerance coding, and TCR marking. [R1] Vishay's manufacturer references
show the band roles and the corresponding digit, multiplier, tolerance, and
optional sixth-band TCR table. [R2][R3]

For the MVP, support only axial banded resistors and use this lookup (normalize
"grey"/"gray" in software):

| Colour | Digit | Multiplier | Tolerance when used as tolerance band |
|---|---:|---:|---:|
| Black | 0 | 10^0 | — |
| Brown | 1 | 10^1 | ±1% |
| Red | 2 | 10^2 | ±2% |
| Orange | 3 | 10^3 | — |
| Yellow | 4 | 10^4 | — |
| Green | 5 | 10^5 | ±0.5% |
| Blue | 6 | 10^6 | ±0.25% |
| Violet | 7 | 10^7 | ±0.1% |
| Grey | 8 | 10^8 | ±0.05% |
| White | 9 | 10^9 | — |
| Gold | — | 10^-1 | ±5% |
| Silver | — | 10^-2 | ±10% |
| No band | — | — | ±20% |

The Vishay four-band diagram is: two significant digits, multiplier,
tolerance. The five-band diagram is: three significant digits, multiplier,
tolerance. A six-band part adds TCR as the sixth band; the first release should
either expose that field or return unsupported_6_band, rather than silently
discarding it. [R2][R3]

For a four-band candidate (d1, d2, m, t):

~~~
significand = 10*d1 + d2
R_nominal_ohms = significand * 10^m
~~~

For a five-band candidate (d1, d2, d3, m, t):

~~~
significand = 100*d1 + 10*d2 + d3
R_nominal_ohms = significand * 10^m
~~~

These equations are the direct decoding of the band roles in the Vishay
references, not a learned regression. The electrical tolerance interval is

~~~
[R_nominal_ohms * (1 - t), R_nominal_ohms * (1 + t)]
~~~

That interval is a specification interval, not a measurement. A visual
decoder should label its result visual_nominal; it cannot claim the actual
resistance without an electrical measurement. If band colours, band count, or
orientation are ambiguous, return the candidate set or abstain instead of
inventing a single Ω value.

### Camera and calibration constraints

OpenCV documents radial and tangential lens distortion, calibration from
known 3-D/2-D points on a pattern, calibrateCamera for camera parameters,
and undistort/rectification for correcting images. [O5][O6] Its color API
also documents that 8-bit BGR-to-HSV uses hue range 0–180, so channel order and
numeric range must be explicit in the implementation. [O4]

Recommended capture contract (engineering recommendations, not claims that
the current repository already satisfies them):

- Fix or record camera identity, focus, exposure, white balance, resolution,
  and distance. Use diffuse, stable illumination and avoid clipped highlights
  on glossy bands.
- Calibrate each supported camera with a known pattern once, undistort before
  geometric cropping, and keep a neutral/colour reference in the capture
  setup if colour repeatability matters.
- Require enough pixels across the body and each band; make minimum-size and
  blur/specular checks explicit quality gates. Do not pick numeric thresholds
  until measured on the validation set.
- Rectify the resistor axis before colour sampling. Sample a band from an
  interior strip or robust median over a region, not one pixel; shadows and
  highlights should lower confidence rather than be treated as new colours.
- Start with a controlled single-resistor crop. Add frame-wide search only
  after the crop decoder is measurable.

## Minimal integration design

~~~
frame/crop
  -> optional existing 20-class classifier gate
  -> body mask + contour/connected-component candidate
  -> axis estimate + undistorted/deskewed body crop
  -> band-count/orientation + colour-band decoder
  -> IEC/Vishay lookup and tolerance interval
  -> confidence, warnings, or explicit abstention
~~~

### Smallest useful first slice

The first prototype should accept a tight frame_bgr resistor crop (or a
user-drawn ROI) on a neutral background and support four-band axial parts.
This isolates the value-recognition question from the harder frame-level
detection question. It should still use the same output contract as the later
camera path.

### Inputs

~~~
frame_bgr: uint8[H,W,3]
roi_xyxy: optional user/primary-model crop
camera_id: optional calibration/colour-profile key
primary_gate: optional {label, confidence, checkpoint_id}
~~~

The existing classifier can be used as a gate only: accept a resistor crop
when its policy says "resistor", or bypass the gate when the user supplied the
ROI. The current checkpoint's 20-class confidence is not a band confidence.

### Outputs

~~~json
{
  "status": "ok|ambiguous|unsupported|no_resistor|calibration_required",
  "component_label": "resistor",
  "component_confidence": 0.0,
  "bbox_xyxy": [0, 0, 0, 0],
  "body_polygon": [],
  "axis_angle_deg": 0.0,
  "band_count": 4,
  "bands": [
    {"slot": 1, "color": "yellow", "confidence": 0.0},
    {"slot": 2, "color": "violet", "confidence": 0.0},
    {"slot": 3, "color": "red", "confidence": 0.0},
    {"slot": 4, "color": "gold", "confidence": 0.0}
  ],
  "nominal_ohms": 4700,
  "tolerance_percent": 5.0,
  "interval_ohms": [4465, 4935],
  "value_source": "visual_nominal",
  "warnings": [],
  "rejection_reason": null
}
~~~

nominal_ohms and interval_ohms should be null unless the standard decoder
has a unique valid interpretation. Keep raw band probabilities, candidate
decodings, geometry quality, and colour-quality statistics in a debug record;
the UI can show the selected bands without hiding why the decoder abstained.

### Processing stages

1. **Gate:** run the current component classifier if a full frame is supplied;
   otherwise require a user ROI. Reject non-resistor top predictions and keep
   a review path for low-confidence cases.
2. **Localize:** build HSV/Lab/body masks; find connected components/contours;
   score candidates by area, elongation, solidity, and lead/body plausibility.
3. **Rectify:** choose the body candidate, estimate its long axis, rotate/crop,
   and remove leads/end margins. If multiple candidates survive, abstain.
4. **Find bands:** form a one-dimensional profile across the long axis using
   robust colour statistics over the body width. Detect contiguous band-like
   runs and estimate band centres/widths. Do not assume four bands when five or
   six are visible.
5. **Decode:** evaluate both directions and plausible four-/five-band layouts,
   map colours through the standard table, and retain all candidates that pass
   geometry and tolerance rules.
6. **Uncertainty:** combine colour separability, band-width/spacing quality,
   orientation agreement, and primary-gate confidence. If one candidate does
   not dominate, return ambiguous with candidates; if the object is not an
   axial banded resistor, return unsupported.
7. **Display:** overlay the body box, axis, band slots, decoded string, nominal
   value, tolerance interval, and any warning. Never display an exact-looking
   value without the tolerance and status.

## When a learned secondary model is justified

### Not justified by the current checkout

The current resistor split has only a coarse class label, 25 validation images,
and 25 test images; it has no code/value/tolerance labels and includes
heterogeneous resistor forms and web-scraped noise. A new supervised band model
would therefore have no valid target and would risk learning dataset/style
artifacts. The existing primary_resnet18.pt is trained for 20-way component
classification, not resistance. Transfer-learning that checkpoint's
architecture does not supply resistance supervision.

### Justified trigger

Train a secondary model only if all of the following hold:

- A measured baseline on representative crops shows repeatable failure after
  colour/camera calibration, for example band boundaries or colour classes are
  the dominant errors.
- The product needs scenes beyond the classical assumptions: cluttered
  backgrounds, partial occlusion, worn/faded bands, or multiple resistor
  instances.
- New labels cover band count, band slot/colour, tolerance, body geometry, and
  a physical component identifier. Splits must be by physical resistor and
  capture session, not by near-duplicate frames.
- The model can abstain and expose calibrated per-band probabilities. The
  standard-constrained decoder remains the source of the Ω calculation.

The minimal learned component should take a rectified body crop plus the body
mask (optionally HSV/Lab channels) and output:

~~~
band_count logits
per-slot colour logits (4/5 slots, with an explicit unknown class)
tolerance-band logits / orientation
quality-or-abstain score
~~~

It should not directly regress Ω from the full image. A shared small CNN is
enough for a first benchmark; if a pretrained/edge-friendly option is needed,
the original MobileNetV3 paper explicitly targets high- and low-resource mobile
variants, while the original ResNet paper describes the 18-layer residual
family already represented by this repository's primary checkpoint. [M1][M2]
Those papers support architecture choices only; they do not support reusing the
current checkpoint for resistance or replacing the standard decoder.

## Validation data and acceptance plan

### New data required

Create a small, purpose-built axial-resistor set before training or claiming
accuracy. At minimum it should include:

- multiple physical specimens for at least ten distinct four-/five-band codes;
  include low/high values and tolerance bands in gold, silver, brown, red, and
  precision colours where available;
- multiple manufacturers/body colours, resistor sizes, orientations, distances,
  backgrounds, illumination levels, focus/blur conditions, and mild occlusion;
- a held-out set of physical specimens and capture sessions, not just new crops
  of the same resistor;
- ground truth for component_id, band count, ordered band colours, tolerance,
  body box/polygon, axis/endpoints, nominal Ω from the marked code, and an
  independent meter reading where electrical accuracy is being assessed.

The existing resistor images may be used for a coarse "does the gate reject
obvious non-axial examples?" smoke test, but they cannot be the resistance
benchmark without new labels. Keep the current component test split separate
from any value-model selection.

### Metrics

Report these separately rather than hiding everything in one accuracy number:

- component-gate precision/recall and false-accept rate;
- body detection/segmentation quality and axis error;
- band count accuracy, per-slot colour accuracy, direction/orientation
  accuracy, exact band-code accuracy, and tolerance accuracy;
- exact nominal-value accuracy for supported parts, plus log10-ohm error only
  as a secondary diagnostic;
- interval coverage: how often the independent electrical reading falls in
  the reported specification interval, with reasons recorded for exceptions;
- selective accuracy/coverage after abstention, confidence calibration, and
  latency on the target camera.

Set promotion thresholds before looking at the held-out set. A model that
returns a confident wrong code is worse than a decoder that abstains on an
unreadable band. Re-run the same physical specimens under a new session to
measure camera/lighting drift.

## Main risks and explicit non-goals

- "Resistor" is broader than "axial colour-banded resistor." SMD numeric marks,
  printed power resistors, resistor networks, shunts, cement/wire-wound parts,
  collages, and a resistor seen through a package require different decoders.
- A colour band is not a pixel-colour lookup. White balance, illuminant, body
  colour, shadows, specular reflections, blur, compression, and adjacent
  bands can change the observed distribution.
- Band direction, four-vs-five-band count, and a sixth TCR band can be
  ambiguous. Enumerate candidates and abstain; never silently reverse a code.
- The visual result is a nominal marked value with a manufacturer/standard
  tolerance. It is not a calibrated resistance measurement, and this MVP does
  not estimate temperature, power derating, ageing, or in-circuit effects.
- There is no current repository basis for a learned resistance model or a
  reliable frame-wide OpenCV demo. The smallest honest milestone is a
  controlled-crop, four-band classical decoder with a validation harness.

## Sources

Only primary/high-trust sources were used for external technical claims:

- **[O1]** OpenCV, "Thresholding Operations using inRange" (official
  documentation): <https://docs.opencv.org/4.x/da/d97/tutorial_threshold_inRange.html>
- **[O2]** OpenCV, "Contours: Getting Started" (official documentation):
  <https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html>
- **[O3]** OpenCV, "Contour Features" and shape API (official documentation):
  <https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html>;
  <https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html>
- **[O4]** OpenCV, color-conversion API (official documentation):
  <https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html>
- **[O5]** OpenCV, "Camera Calibration" (official documentation):
  <https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html>
- **[O6]** OpenCV, calib3d API (official documentation):
  <https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html>
- **[O7]** OpenCV, geometric image transformations API (official
  documentation): <https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html>
- **[R1]** IEC, "IEC 60062:2016 — Marking codes for resistors and capacitors"
  (official standard description):
  <https://webstore.iec.ch/en/publication/25395>
- **[R2]** Vishay Intertechnology, "Color Code and Standard Resistance Series"
  (manufacturer reference, PDF):
  <https://www.vishay.com/docs/20143/colorcod.pdf>
- **[R3]** Vishay Intertechnology, "Resistor Color Code Calculator" (manufacturer
  reference, PDF):
  <https://www.vishay.com/docs/49411/resistor_color_code_calculator.pdf>
- **[M1]** He et al., "Deep Residual Learning for Image Recognition," original
  paper: <https://arxiv.org/abs/1512.03385>
- **[M2]** Howard et al., "Searching for MobileNetV3," original paper:
  <https://arxiv.org/abs/1905.02244>

