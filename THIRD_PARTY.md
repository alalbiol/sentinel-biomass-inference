# Sources and third-party components

- **AI4DIS inference architecture:** extracted from `use_case_1/models.py` of the original project. The computation and parameter names are retained for compatibility. The lightweight loader replaces the Lightning training wrapper.
- **PyTorch / torchvision:** installed dependencies; retain their upstream terms.
- **timm:** installed dependency providing encoder implementations; [source](https://github.com/huggingface/pytorch-image-models).
- **segmentation_models_pytorch:** installed dependency providing the U-Net decoder; [source](https://github.com/qubvel-org/segmentation_models.pytorch).
- **BioMassters:** original training-data source, not bundled here; [dataset](https://huggingface.co/datasets/nascetti-a/BioMassters) and [paper](https://nascetti-a.github.io/BioMasster/).
- **Copernicus Sentinel sample imagery:** original/modified source data retain the Copernicus terms. See samples/README.md for attribution and processing.
- **Earth Engine:** optional external service requiring the user's own authentication and an enabled project. The downloader uses the official Python API.

No third-party training dataset, paper manuscript, credentials, API token, or Google project configuration is included in the snapshot.
