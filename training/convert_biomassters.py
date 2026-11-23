"""Convert BioMassters GeoTIFF features to one safetensors file per chip (the training input format).

    python -m training.convert_biomassters --features-dir data/biomassters/train_features \
        --output-dir data/biomassters/train_safetensors --workers 8

Each <chip_id>.safetensors holds `features` (float32 [12, 15, 256, 256], raw values) and
`missing` (int64 indices of months without Sentinel-2). Values are those of
training.data.read_chip_tifs, so training from the converted files or from the GeoTIFFs is the
same. Converted chips take about 47 MB each (about 410 GB for the 8,689 training chips), against
216 GB of GeoTIFFs, but load much faster. Existing files are skipped.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from safetensors.numpy import save_file

from training.data import read_chip_tifs


def convert(chip_id, features_dir, output_dir):
    target = output_dir / f'{chip_id}.safetensors'
    if target.exists():
        return
    features, missing = read_chip_tifs(chip_id, features_dir)
    tmp = target.with_suffix('.tmp')
    save_file({'features': features, 'missing': missing}, str(tmp))
    tmp.replace(target)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--features-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    chips = sorted({f.name.split('_')[0] for f in args.features_dir.glob('*_S1_00.tif')})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f'{len(chips)} chips', flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        for i, _ in enumerate(pool.map(convert, chips, [args.features_dir] * len(chips),
                                       [args.output_dir] * len(chips), chunksize=8), 1):
            if i % 500 == 0 or i == len(chips):
                print(f'{i}/{len(chips)}', flush=True)


if __name__ == '__main__':
    main()
