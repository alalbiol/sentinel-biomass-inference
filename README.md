---
name: sentinel-biomass-inference
authors:
  - Alberto Albiol
  - Miquel Obrador-Reina
  - Juan B. Pedro
  - Albert Fernández
license: MIT
source: https://github.com/alalbiol/sentinel-biomass-inference/tree/v1.0.0
---

# Sentinel biomass inference

Estimate aboveground dry biomass from a year of Sentinel-1 and Sentinel-2 images using the Finland-trained AI4DIS models. Outputs are single-band biomass maps in **Mg/ha**, at a 10 m grid spacing.

**Release:** [GitHub v1.0.0](https://github.com/alalbiol/sentinel-biomass-inference/releases/tag/v1.0.0).
**EOTDL model snapshot:** prepared for upload as `sentinel-biomass-inference`; the [intended model page](https://www.eotdl.com/models/sentinel-biomass-inference) will become available after publication. Its EOTDL version number has not yet been assigned.

GitHub contains the inference code and two real example patches. The EOTDL snapshot additionally contains all ten checkpoints, with the exact same code and examples. This release estimates standing biomass; it does not forecast future growth or convert biomass into carbon credits.

## Quick start

Use Python 3.11. Create an environment and install the dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
# CPU installation; for CUDA use the appropriate PyTorch wheel instead.
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Download the complete EOTDL snapshot once it is published, or obtain its `checkpoints/` folder and place it at the repository root. Checkpoints are not stored in Git. `provenance/checkpoints.json` records their names, sizes, and hashes.

```bash
python scripts/verify_checkpoints.py
python inference.py \
  --model mobilenetv4_early \
  --input-dir samples/inputs \
  --chip-id ES_2021_42p9492_W3p2578 \
  --output outputs/biomass.tif \
  --device cpu
```

`outputs/biomass.tif` retains the input CRS and grid. A companion JSON records the model, weight checksum, missing optical months, and mean prediction. Negative predictions are retained to match the original models; no clipping or post-processing is applied.

Check a result against the stored project predictions:

```bash
python scripts/verify_samples.py --models mobilenetv4_early --device cpu
```

This tests numerical agreement with earlier model outputs, not accuracy against field measurements. Both bundled patches are from Spain, outside the training domain. Read [MODEL_CARD.md](MODEL_CARD.md) before interpreting predictions.

## Available models

Five backbones are provided with `early` and `late` temporal fusion:

- `convnextv2_early`, `convnextv2_late`
- `efficientnet_early`, `efficientnet_late`
- `efficientnetv2_early`, `efficientnetv2_late`
- `mobilenetv4_early`, `mobilenetv4_late`
- `resnetrs101_early`, `resnetrs101_late`

The ten inference-only safetensors files total approximately 2.1 GiB. They contain the original model parameters without Lightning training state or optimizer tensors. Loading is strict and does not download pretrained backbone weights. MobileNetV4 is the small default for trying the code, not a claim that it is the most accurate model.

## Input contract

Each patch has **256×256 pixels**, **12 monthly positions**, and 15 channels per month. Use co-registered GeoTIFFs named:

```text
{chip_id}_S1_00.tif ... {chip_id}_S1_11.tif
{chip_id}_S2_00.tif ... {chip_id}_S2_11.tif
```

| Sensor | Bands, in this exact order | Values |
|---|---|---|
| Sentinel-1 | VV ascending, VH ascending, VV descending, VH descending | Backscatter in dB; `-9999` is replaced by zero |
| Sentinel-2 | B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, CLP | Raw reflectance DN (approximately reflectance × 10000); cloud probability on 0–255 scale |

**Do not normalize or divide the optical DN by 10000 before inference.** The original checkpoints were evaluated with raw inputs. No normalization option is exposed.

All twelve radar files are required. An unavailable optical month is represented by an absent S2 file and a missing-month mask. At least one optical month must exist. Invalid shapes, misaligned grids, and non-finite values are rejected. Channel names are positional; the loader does not infer band identities from pixel values.

## Optional: download your own patch from Earth Engine

Ready-to-run samples do not require Earth Engine. To acquire a new patch, install the optional dependencies and authenticate with your own Earth Engine account and project:

```bash
python -m pip install -r requirements-gee.txt
earthengine authenticate
export GEE_PROJECT="your-earth-engine-enabled-project"
python scripts/download_gee.py \
  --lon -3.25776 --lat 42.94916 \
  --start-date 2020-09-01 \
  --chip-id my_patch \
  --output-dir downloads/my_patch

python inference.py \
  --input-dir downloads/my_patch --chip-id my_patch \
  --output outputs/my_patch_biomass.tif
```

The downloader produces one 2.56 km square on a common 10 m local UTM grid and records acquisition dates and source scenes. `--start-date` is the first day of a month; twelve months are downloaded. September–August matches the training dataset’s seasonal window.

The default `biomassters-style` recipe uses monthly S1 means per orbit, masking backscatter below −25 dB, and the full-coverage S2 scene with the lowest mean s2cloudless probability each month. Cloud probability is scaled from 0–100 to 0–255. The `spain-evaluation` recipe uses the per-orbit S1 medians used in the Spain case study. The bundled examples use that recipe and January–December 2021. These GEE recipes approximate the feature construction; they do not recreate the exact original BioMassters acquisition system or establish predictive accuracy for a new area.

Downloads resume only when their saved acquisition configuration agrees with the requested patch. A month with no suitable S2 scene is recorded as missing. `--crs` can select another projected CRS in metres. `--months 1` is for download diagnostics only; inference requires twelve radar months.

## Repository contents

```text
biomass/       Network architecture, input handling, and inference
inference.py   Inference command
samples/       Two real multiband inputs and stored reference predictions
scripts/       GEE acquisition, verification, and snapshot preparation
tests/         Input validation tests
provenance/    Checkpoint checksums, origins, and validation results
```

See [EOTDL.md](EOTDL.md) for preparing and uploading the model snapshot, and [samples/README.md](samples/README.md) for the sample provenance. Tests run with `python -m unittest discover -s tests`.

## Authors, licence, and acknowledgement

Alberto Albiol, Miquel Obrador-Reina, Juan B. Pedro, and Albert Fernández.

Project code and released model weights use the [MIT licence](LICENSE). Third-party libraries retain their respective licences. Sample imagery contains modified Copernicus Sentinel data (2021) and retains the Copernicus source terms; it is not relicensed as MIT. See [THIRD_PARTY.md](THIRD_PARTY.md).

Prepared within AI4DIS, funded by the European Space Agency under Contract No. 4000147555/25/I-DT. The Universitat Politècnica de València participated as a subcontractor to EarthPulse S.L., the prime contractor. The views expressed are those of the authors and do not necessarily reflect the official position of the European Space Agency.
