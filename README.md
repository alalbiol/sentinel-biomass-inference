---
name: sentinel-biomass-inference
authors:
  - Alberto Albiol
  - Miquel Obrador-Reina
  - Juan B. Pedro
  - Albert Fernández
license: MIT
source: https://github.com/alalbiol/sentinel-biomass-inference/tree/v1.1.0
---

# Sentinel biomass inference

**Ten deep-learning models that map aboveground forest biomass (Mg/ha, 10 m pixels) from a year of Sentinel-1 radar and Sentinel-2 optical images.** They were trained by the AI4DIS project on the Finnish [BioMassters](https://huggingface.co/datasets/nascetti-a/BioMassters) benchmark, with five encoder backbones and two ways of combining the twelve months. This repository holds everything needed to use them and to rebuild them:

- **run** a model on a 2.56 × 2.56 km patch, with two real example patches and their reference outputs;
- **download** new patches from Google Earth Engine;
- **retrain** any of the ten models from BioMassters with the original settings, and **score** it on the BioMassters test set.

> **The model weights are on EOTDL, not on GitHub.** Get them from the **[EOTDL model `sentinel-biomass-inference`](https://www.eotdl.com/models/sentinel-biomass-inference)** (2.1 GB, ten `.safetensors` files). Each EOTDL version is a complete snapshot: this code plus `checkpoints/`.

| Release | GitHub | EOTDL | Contents |
|---|---|---|---|
| 1.0.0 | [v1.0.0](https://github.com/alalbiol/sentinel-biomass-inference/releases/tag/v1.0.0) | [model version 1](https://www.eotdl.com/models/sentinel-biomass-inference) | Inference, examples, Earth Engine downloader, ten checkpoints |
| 1.1.0 | [v1.1.0](https://github.com/alalbiol/sentinel-biomass-inference/releases/tag/v1.1.0) | model version 2 (pending upload) | Adds the training and evaluation code, recovered training settings and BioMassters test results. Same checkpoints. |

The models estimate standing biomass. They do not forecast growth or compute carbon credits, and they were trained in boreal Finland: read [MODEL_CARD.md](MODEL_CARD.md) before using them elsewhere.

## The models

| Encoder (timm weights it started from) | Early fusion: RMSE / R² | Late fusion: RMSE / R² | Weights (each) |
|---|---|---|---|
| ConvNeXt V2 Base (`convnextv2_base.fcmae_ft_in22k_in1k`) | 34.99 / 0.772 | 39.09 / 0.716 | 377 MB |
| EfficientNet-B5 (`tf_efficientnet_b5.ns_jft_in1k`) | **33.30 / 0.794** | 39.47 / 0.710 | 125 MB |
| EfficientNetV2-M (`tf_efficientnetv2_m.in21k_ft_in1k`) | 34.52 / 0.778 | 38.56 / 0.723 | 225 MB |
| MobileNetV4 Conv Medium (`mobilenetv4_conv_medium.e500_r224_in1k`) | 35.27 / 0.769 | 42.40 / 0.665 | 49 MB |
| ResNet-RS-101 (`resnetrs101.tf_in1k`) | 33.98 / 0.785 | 39.79 / 0.705 | 308 MB |
| Ensemble of the five | 33.56 / 0.791 | 39.31 / 0.713 | |

BioMassters test set: 2,773 chips in Finland, airborne-lidar biomass labels, metrics pooled over all 10-m pixels (RMSE in Mg/ha). Full metrics: [expected/biomassters_test.json](expected/biomassters_test.json); `training/evaluate.py` recomputes them.

Model ids are `<encoder>_<fusion>`: `convnextv2`, `efficientnet`, `efficientnetv2`, `mobilenetv4`, `resnetrs101`, each with `early` or `late`. MobileNetV4 early fusion is the default because it is small, not because it is the most accurate.

**How they work.** All ten share one network, a U-Net with a [timm](https://github.com/huggingface/pytorch-image-models) encoder (`biomass/model.py`). Each month (15 channels: 4 Sentinel-1 and 11 Sentinel-2 bands) goes through the encoder separately.
- **Early fusion:** a learned attention weights the twelve months at every encoder scale before one decoder produces the map.
- **Late fusion:** every month is decoded on its own and the maps of the months with Sentinel-2 data are averaged. During training each chip contributes one randomly chosen month per step.

## Quick start: run a model

Use Python 3.11:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
# CPU installation; for CUDA use the matching PyTorch wheel instead.
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Get the checkpoints from EOTDL (the [EOTDL client](https://www.eotdl.com/docs/getting-started/install) needs Python 3.12+, so keep it in its own environment):

```bash
eotdl auth login
eotdl models get sentinel-biomass-inference --path ./downloads
cp -r ./downloads/<downloaded folder>/checkpoints .     # or work inside the downloaded snapshot
python scripts/verify_checkpoints.py                   # sizes and SHA-256 against provenance/checkpoints.json
```

Run a model on an example patch and compare with the stored reference output:

```bash
python inference.py --model mobilenetv4_early --input-dir samples/inputs \
  --chip-id ES_2021_42p9492_W3p2578 --output outputs/biomass.tif --device cpu
python scripts/verify_samples.py --models mobilenetv4_early --device cpu
```

`outputs/biomass.tif` keeps the input CRS and grid; a JSON next to it records the model, the weight checksum, the missing optical months and the mean prediction. Negative predictions are kept, as in the original models. The two example patches are from Spain, outside the training domain; the check is numerical agreement with the original outputs, not accuracy.

## Input format

Each patch has **256 × 256 pixels** at 10 m, **12 monthly positions** and 15 channels per month, as co-registered GeoTIFFs:

```text
{chip_id}_S1_00.tif ... {chip_id}_S1_11.tif
{chip_id}_S2_00.tif ... {chip_id}_S2_11.tif
```

| Sensor | Bands, in this order | Values |
|---|---|---|
| Sentinel-1 | VV ascending, VH ascending, VV descending, VH descending | Backscatter in dB; `-9999` (no pass) becomes zero |
| Sentinel-2 | B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, CLP | Raw L2A digital numbers (about reflectance × 10,000); cloud probability on 0–255 |

**Do not normalize the inputs.** The models were trained and evaluated on raw values. All twelve radar files are required. A month without an optical image has no S2 file and is masked; at least one optical month must exist. Wrong shapes, misaligned grids and non-finite values are rejected.

## Download a patch from Google Earth Engine

```bash
python -m pip install -r requirements-gee.txt
earthengine authenticate
export GEE_PROJECT="your-earth-engine-enabled-project"
python scripts/download_gee.py --lon -3.25776 --lat 42.94916 --start-date 2020-09-01 \
  --chip-id my_patch --output-dir downloads/my_patch
python inference.py --input-dir downloads/my_patch --chip-id my_patch --output outputs/my_patch_biomass.tif
```

The downloader writes one 2.56-km square on a 10-m local UTM grid, with twelve months from `--start-date` (September to August matches the training data), and records acquisition dates and source scenes.
- The default `biomassters-style` recipe uses monthly Sentinel-1 means per orbit (backscatter below −25 dB masked) and, each month, the Sentinel-2 scene covering the patch with the lowest mean s2cloudless probability, scaled to 0–255.
- `spain-evaluation` uses per-orbit Sentinel-1 medians, as in the AI4DIS Spain case study.

Both recipes approximate how BioMassters was built; they do not reproduce it exactly. `--crs` selects another metric CRS, and downloads resume when the saved configuration matches.

## Reproduce the training

`training/` contains the code that trained the released models, ported from the AI4DIS project with the same network, loss, optimizer and schedule. The defaults are the settings of the released checkpoints, read from the training state that Lightning stored in each of them ([provenance/training.json](provenance/training.json)):

| Setting | Value |
|---|---|
| Data | BioMassters training split, 8,689 chips; random 80% (6,951) for fitting, 20% for validation |
| Inputs | Raw values (see the input format), no normalization, no augmentation |
| Encoder initialization | timm ImageNet weights (table above) |
| Loss | Mean squared error over the 10-m pixels |
| Optimizer | AdamW, learning rate 1e-3, weight decay 0.01 |
| Schedule | Cosine annealing from 1e-3 to 1e-6 over 20 epochs, stepped per epoch |
| Batch | 3 chips (36 monthly frames), one GPU, 32-bit with TF32 matrix products |
| Length | 20 epochs (46,340 steps); the checkpoint of the last epoch is kept |

**1. Install** the training extras (Lightning 2.6.1, as in the original runs), in the environment above or a CUDA one:

```bash
python -m pip install -r requirements-train.txt
```

**2. Get BioMassters** from [Hugging Face](https://huggingface.co/datasets/nascetti-a/BioMassters) (CC BY 4.0). The training features come as a split ZIP archive (`train_features.zip` with `.z01`–`.z13`, 216 GB unpacked), and the labels as `train_agbm.zip` (8,689 `<chip_id>_agbm.tif`). The test set is `test_features_splits.*` (73 GB) and `test_agbm.tar`.

```bash
python -m pip install huggingface_hub           # provides the `hf` command
hf download nascetti-a/BioMassters --repo-type dataset --local-dir data/biomassters
cd data/biomassters
zip -s 0 train_features.zip --out train_features_full.zip
unzip -q -j train_features_full.zip -d train_features   # -j: the archive stores files under a long folder path
unzip -q train_agbm.zip                                   # -> train_agbm/
zip -s 0 test_features_splits.zip --out test_features_full.zip && unzip -q -j test_features_full.zip -d test_features
tar xf test_agbm.tar
cd ../..
```

The download is about 300 GB; add `--include "train_*"` (or `"test_*"`) to fetch one split.

**3. Convert** each chip's 24 GeoTIFFs to one file, so that loading keeps the GPU busy (about 47 MB per chip, 410 GB for the training split):

```bash
python -m training.convert_biomassters --features-dir data/biomassters/train_features \
  --output-dir data/biomassters/train_safetensors
```

With `--format tif`, `train.py` and `evaluate.py` read the GeoTIFFs directly instead; the arrays are identical, only slower.

**4. Train** one model, or all ten with `training/run_all.sh`:

```bash
python -m training.train --model mobilenetv4_early \
  --features-dir data/biomassters/train_safetensors --labels-dir data/biomassters/train_agbm
```

This writes `runs/<model>/`: the Lightning checkpoint, CSV logs of the training and validation RMSE per epoch, and the inference weights `<model>.safetensors`, which load into `inference.py` with `--model-dir runs/<model>`.

How long and how much GPU memory, measured on one 12 GB NVIDIA RTX 3060 at batch size 3 (GPU time only; keep the converted chips on an SSD, since every step reads about 140 MB):

| Encoder | 32-bit (original) | `--precision bf16-mixed` |
|---|---|---|
| MobileNetV4 | 1.6–1.8 h for 20 epochs, 3.7 GB | not needed |
| ResNet-RS-101 | 7–8 h, 11.7 GB | not needed |
| EfficientNet-B5, EfficientNetV2-M | do not fit in 12 GB | 3.6–3.8 h, 8–9 GB |
| ConvNeXt V2 Base | do not fit in 12 GB | do not fit in 12 GB |

The original runs used a larger GPU; with 24 GB or more, every model trains at the original 32-bit precision. Mixed precision is a departure from the recipe, so expect slightly different results.

**5. Score** it on the BioMassters test set, after converting `test_features` the same way:

```bash
python -m training.evaluate --models mobilenetv4_early --model-dir runs/mobilenetv4_early \
  --features-dir data/biomassters/test_safetensors --labels-dir data/biomassters/test_agbm
python -m training.evaluate --ensemble early --features-dir ... --labels-dir ...   # five-encoder ensemble
```

Without `--model-dir`, the released checkpoints in `checkpoints/` are scored. For the released `mobilenetv4_early`, this reproduces the original results to every reported decimal (RMSE 35.2710, MAE 19.0621, R² 0.7685).

**What will differ.** A retrained model will be close to, but not identical with, the released one:
- the original 80/20 split was drawn without a seed (here `--seed`, default 0, fixes it);
- GPU training is not bit-deterministic;
- the encoder initialization is inferred from the code history (`provenance/training.json`), because the checkpoints do not record it.

Expect test metrics near the table above rather than equal to them.

## Repository contents

```text
biomass/        Network (model.py), input reading, inference
inference.py    Inference command
samples/        Two real Spanish patches and the reference outputs of all ten models
scripts/        Earth Engine download, checkpoint and sample verification, snapshot preparation
training/       BioMassters conversion, training, export, test-set evaluation, run_all.sh
expected/       Original BioMassters test results (Table I of the accompanying study)
provenance/     Checkpoint hashes and origin, recovered training settings, validation records, releases
tests/          Input contract and training-code tests: python -m unittest discover -s tests
```

[EOTDL.md](EOTDL.md) explains how the EOTDL snapshot is built and published, and [samples/README.md](samples/README.md) where the example patches come from.

## Authors, licence, citation and acknowledgement

Alberto Albiol, Miquel Obrador-Reina, Juan B. Pedro and Albert Fernández. The training pipeline was originally written in the AI4DIS project at the Universitat Politècnica de València.

Code and model weights: [MIT licence](LICENSE). Third-party libraries keep their licences, and the example imagery contains modified Copernicus Sentinel data (2021) under the Copernicus terms ([THIRD_PARTY.md](THIRD_PARTY.md)). If you use the models, please also cite BioMassters:

> A. Nascetti *et al.*, "BioMassters: A Benchmark Dataset for Forest Biomass Estimation using Multi-modal Satellite Time-series," *NeurIPS Datasets and Benchmarks Track*, 2023.

The models are evaluated for transfer to Spain in a study under review; its reproduction package is [spain-biomass-transfer-igarss](https://github.com/alalbiol/spain-biomass-transfer-igarss). The article will be linked here after acceptance.

Prepared within AI4DIS, funded by the European Space Agency under Contract No. 4000147555/25/I-DT. The Universitat Politècnica de València participated as a subcontractor to EarthPulse S.L., the prime contractor. The views expressed are those of the authors and do not necessarily reflect the official position of the European Space Agency.
