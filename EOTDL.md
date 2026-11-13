# Model snapshot publication

This package is an EOTDL **model entry** named `sentinel-biomass-inference`.
The root README has EOTDL metadata. Code and model weights use MIT; sample imagery retains Copernicus terms. All four authors are listed in README.md.

## GitHub and EOTDL version mapping

GitHub release `v1.0.0` identifies the code and examples. The local EOTDL snapshot additionally contains all ten inference-only checkpoints. `release-manifest.json` records its source commit, release tag, and per-file hashes.

The EOTDL version is **pending upload**. Its number must be read from EOTDL after publication, then explicitly linked to GitHub `v1.0.0`. Do not assume that it is assigned automatically from the Git tag.

## Rebuild the snapshot

A clean checkout of the `v1.0.0` tag and the released `checkpoints/` files are required:

```bash
python scripts/verify_checkpoints.py
python scripts/prepare_snapshot.py
```

The snapshot is written to `dist/v1.0.0/sentinel-biomass-inference/`. Existing output folders are not overwritten. It contains independently copied weights and files, not symlinks to your working directory.

To reproduce weight conversion from the original internal Lightning checkpoint directory:

```bash
python scripts/export_checkpoints.py --source /path/to/original/checkpoints
```

The source checkpoints are not needed for inference and are not uploaded to GitHub.

## Upload

Use a separate Python 3.12+ environment for the EOTDL CLI:

```bash
conda activate eotdl
eotdl auth login
eotdl models ingest -p ./dist/v1.0.0/sentinel-biomass-inference
```

After upload, verify the public [model page](https://www.eotdl.com/models/sentinel-biomass-inference), record its version and URL, and add reciprocal links to the GitHub release notes and EOTDL description. The GitHub repository website can then point to this model page. Local snapshot preparation does not publish to EOTDL.

## Download after publication

In an environment with the EOTDL client, replace `VERSION_FROM_EOTDL` with the actual published integer:

```bash
eotdl auth login
eotdl models get sentinel-biomass-inference --version VERSION_FROM_EOTDL --path ./downloaded-model
```

Use the resulting snapshot's code and its `checkpoints/` folder together. Verify checkpoint hashes before running inference. For EOTDL client versions whose download command fails, the model page is the alternative entry point; do not silently substitute a different model version.

[Official model ingestion instructions](https://www.eotdl.com/docs/models/ingest)
