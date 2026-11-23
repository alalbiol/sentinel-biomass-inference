# Sources and third-party components

- **AI4DIS architecture and training pipeline:** `biomass/model.py` is `use_case_1/models.py` of the original project, unchanged. `training/` ports its Lightning training module, data loading and evaluation (`utils.py`, `dataset.py`, `train.py`, `test.py`, `test_ensemble.py`), keeping the computation and parameter names.
- **Lightning, scikit-image, pandas:** training and evaluation dependencies; retain their upstream terms.
- **PyTorch / torchvision:** installed dependencies; retain their upstream terms.
- **timm:** installed dependency providing encoder implementations; [source](https://github.com/huggingface/pytorch-image-models).
- **segmentation_models_pytorch:** installed dependency providing the U-Net decoder; [source](https://github.com/qubvel-org/segmentation_models.pytorch).
- **BioMassters:** training and test data (CC BY 4.0), not bundled here (download instructions in README.md); [dataset](https://huggingface.co/datasets/nascetti-a/BioMassters) and [paper](https://nascetti-a.github.io/BioMasster/).
- **Copernicus Sentinel sample imagery:** original/modified source data retain the Copernicus terms. See samples/README.md for attribution and processing.
- **Earth Engine:** optional external service requiring the user's own authentication and an enabled project. The downloader uses the official Python API.

No third-party training dataset, paper manuscript, credentials, API token, or Google project configuration is included in the snapshot.
