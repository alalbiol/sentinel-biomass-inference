# Model snapshot publication

This package is an EOTDL **model entry** named `sentinel-biomass-inference`.
The root README has EOTDL metadata. Code and model weights use MIT; sample imagery retains Copernicus terms. All four authors are listed in README.md.

## Published versions

GitHub release `v1.1.0` adds the training and evaluation code; its snapshot contains the same ten checkpoints and becomes EOTDL model version 2 when uploaded.

GitHub release `v1.0.0` (commit `fa746faefbf6fa9a26339b3fa28c2b8fa7b56752`) corresponds to EOTDL model version **1** (ID `6af713a1dbc0af5e134f7df9`), uploaded on 13 November 2026. The EOTDL catalog checksum of every one of the 105 payload files, and the uploaded `release-manifest.json` itself, were checked against the snapshot built from that commit on 17 November 2026. See `provenance/releases/v1.0.0.json`. Changes on `main` after this tag are not part of the upload; a new upload creates a new EOTDL version.

## Rebuild the snapshot

A clean checkout of the release tag (the version in `VERSION`) and the released `checkpoints/` files are required:

```bash
python scripts/verify_checkpoints.py
python scripts/prepare_snapshot.py
```

The snapshot is written to `dist/v<VERSION>/sentinel-biomass-inference/`. Existing output folders are not overwritten. It contains independently copied weights and files, not symlinks to your working directory.

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
eotdl models ingest -p ./dist/v1.1.0/sentinel-biomass-inference
```

After upload, verify the public [model page](https://www.eotdl.com/models/sentinel-biomass-inference), record its version and URL, and add reciprocal links to the GitHub release notes and EOTDL description. The GitHub repository website can then point to this model page. Local snapshot preparation does not publish to EOTDL.

## Download after publication

In an environment with the EOTDL client (omit `--version` for the latest):

```bash
eotdl auth login
eotdl models get sentinel-biomass-inference --version 1 --path ./downloaded-model
```

Use the resulting snapshot's code and its `checkpoints/` folder together. Verify checkpoint hashes before running inference. For EOTDL client versions whose download command fails, the model page is the alternative entry point; do not silently substitute a different model version.

[Official model ingestion instructions](https://www.eotdl.com/docs/models/ingest)
