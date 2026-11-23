# AI4DIS Sentinel biomass model card — v1.1.0

## Task and scope

Pixel-level regression of aboveground dry biomass in Mg/ha from twelve months of Sentinel-1 radar and Sentinel-2 optical imagery. Pixel spacing is 10 m; this does not imply independent field validation at 10 m or effective physical resolution at that scale.

The models were trained on the Finland BioMassters task. Five encoder backbones are released, each with early attention aggregation and late decoder-feature aggregation. All checkpoints share the UnetVFLOW implementation in `biomass/model.py`. Model weights are loaded strictly into the architecture; no additional fine-tuning is performed here.

## Source data

BioMassters contains forest patches in Finland with LiDAR-calibrated biomass labels and monthly Sentinel images. Its original acquisition window runs from September of the previous year through August of the reference year. The BioMassters source card specifies the band order used by this release. See:

- [BioMassters dataset and paper](https://nascetti-a.github.io/BioMasster/)
- [BioMassters dataset card and data licence](https://huggingface.co/datasets/nascetti-a/BioMassters)

This repository does not redistribute the training dataset. Since v1.1.0 it includes the training code (`training/`), ported from the original AI4DIS pipeline, and an evaluation script for the BioMassters test set. The released checkpoints are the originals; they were not retrained for this release.

## Training

All ten models were trained the same way. The settings below come from the training state stored in each original checkpoint (`provenance/training.json`):
- data: 80% of the 8,689 BioMassters training chips (random split, unseeded), with 20% for validation;
- loss and optimization: pixel-wise MSE on raw inputs, AdamW (learning rate 1e-3, weight decay 0.01), cosine annealing to 1e-6;
- length and batch: 20 epochs, batch size 3 on one GPU;
- checkpoint: the last epoch.

The encoders started from timm ImageNet weights. This is inferred from the code history, because the checkpoints do not record it. Late-fusion models train on one randomly chosen available month per chip and step, and average the decoded months at inference.

## Evaluation on BioMassters

On the 2,773 test chips (native 10 m, pixel-pooled), RMSE ranges from 33.30 Mg/ha (EfficientNet-B5, early fusion, R² 0.794) to 42.40 Mg/ha (MobileNetV4, late fusion, R² 0.665). Early fusion beats late fusion for every encoder. All values are in `expected/biomassters_test.json`, and `training/evaluate.py` recomputes them from the checkpoints. These are in-domain results in Finland; they do not establish accuracy elsewhere.

## Preprocessing and missing observations

See README.md for the exact band order, file layout, and raw units. Sensor values are not normalized. Radar sentinel values of −9999 become zero. Missing optical files become zero-valued optical channels and set the missing mask. The original early-fusion implementation accepts this mask but does not explicitly mask its attention logits; the late-fusion evaluation excludes missing optical months from its temporal mean. This behaviour is preserved for checkpoint compatibility.

The optional Earth Engine downloader is a documented new acquisition recipe. GEE S1 GRD uses dB values; harmonized S2 surface-reflectance DN remain at their native scale. Reprojection and compositing choices can affect predictions. The new default recipe is similar to BioMassters, not a claim of byte-equivalence or validated transfer to arbitrary locations.

## Verification

`provenance/validation.json` records comparisons for all ten models on both bundled patches, using original AI4DIS outputs as references. `samples/expected/` holds the complete float32 reference maps. The sample checker uses `rtol=1e-3` with `atol=0.1 Mg/ha` on CUDA and `atol=0.25 Mg/ha` on CPU. The CPU threshold was set after measuring a maximum difference of about 0.22 Mg/ha for the default MobileNetV4 model, compared with stored CUDA outputs; this is a numerical tolerance, not an accuracy claim. All ten models were checked on CUDA; the default MobileNetV4 early-fusion model was also checked on CPU. Values are not clipped.

A small numerical difference from stored predictions demonstrates implementation agreement. It is not a biomass-accuracy result. The sample bundle deliberately contains no field or CCI reference labels.

## Limitations

- Finland-trained models may be biased outside boreal Finland; Spain examples illustrate operation under domain shift.
- Cloud cover, missing observations, sensor units, and compositing choices can materially affect predictions.
- Fixed 256×256 patches are supported; large-area tiling and overlap blending are not implemented.
- There is no calibrated uncertainty output, future growth forecast, root-biomass estimate, or carbon-credit calculation.
- Fine-grained geographic and temporal validation should precede use in a new application.
- Checkpoints can produce negative regression values. Keeping them preserves the original numerical behaviour; downstream handling is an application decision.

## Checkpoint provenance

`provenance/checkpoints.json` records source Lightning checkpoint hashes, the safetensors hashes, and file sizes. `scripts/export_checkpoints.py` removed the common `model.` state prefix and exported every model tensor unchanged. Optimizer and trainer state were omitted. Source files were read with PyTorch `weights_only=True`.
