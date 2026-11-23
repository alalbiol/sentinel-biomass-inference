"""BioMassters inputs for training and evaluation.

Inputs are the raw BioMassters values, exactly as the models were trained: Sentinel-1 in dB
(-9999 no-data set to 0) and Sentinel-2 digital numbers with the s2cloudless band (0-255).
No normalization is applied. A month without a Sentinel-2 file gets zero optical channels and
is listed in `missing`. Each chip is a [12, 15, 256, 256] float32 array (months x channels).

Reading the 24 GeoTIFFs of a chip is slow, so the original training read one pre-converted
safetensors file per chip (see convert_biomassters.py). Both formats give identical arrays.
"""
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import load_file
from skimage import io

IMG_SIZE = (256, 256)


def read_chip_tifs(chip_id, features_dir):
    """The 12 monthly S1/S2 GeoTIFFs of a BioMassters chip -> (features [12, 15, 256, 256], missing month indices)."""
    features_dir = Path(features_dir)
    frames, missing = [], []
    for month in range(12):
        s1 = io.imread(features_dir / f'{chip_id}_S1_{month:02d}.tif')
        s1 = np.where(s1 == -9999, 0, s1.astype('float32'))
        s2_path = features_dir / f'{chip_id}_S2_{month:02d}.tif'
        if s2_path.is_file():
            s2 = io.imread(s2_path).astype('float32')
        else:
            missing.append(month)
            s2 = np.zeros(IMG_SIZE + (11,), dtype='float32')
        frames.append(np.transpose(np.concatenate([s1, s2], axis=2), (2, 0, 1)))
    return np.stack(frames).astype('float32'), np.array(missing, dtype=np.int64)


def read_label(chip_id, labels_dir):
    return io.imread(Path(labels_dir) / f'{chip_id}_agbm.tif').astype('float32')


class BioMassters(torch.utils.data.Dataset):
    """(features, missing mask [12] bool, biomass label [256, 256]) per chip."""

    def __init__(self, chip_ids, features_dir, labels_dir, fmt='safetensors'):
        if fmt not in ('safetensors', 'tif'):
            raise ValueError(fmt)
        self.chip_ids = list(chip_ids)
        self.features_dir, self.labels_dir, self.fmt = Path(features_dir), Path(labels_dir), fmt

    def __len__(self):
        return len(self.chip_ids)

    def __getitem__(self, index):
        chip_id = self.chip_ids[index]
        if self.fmt == 'safetensors':
            data = load_file(str(self.features_dir / f'{chip_id}.safetensors'))
            features, missing = data['features'], data['missing']
        else:
            features, missing = read_chip_tifs(chip_id, self.features_dir)
        mask = torch.zeros(12, dtype=torch.bool)
        mask[torch.from_numpy(np.asarray(missing, dtype=np.int64))] = True
        return torch.from_numpy(features).float(), mask, torch.from_numpy(read_label(chip_id, self.labels_dir))


def list_chips(labels_dir, metafile=None):
    """Chip ids from a CSV with a chip_id column, or from the *_agbm.tif labels present."""
    if metafile is not None:
        import pandas as pd
        return sorted(pd.read_csv(metafile).chip_id.astype(str).unique())
    return sorted(p.name[:-len('_agbm.tif')] for p in Path(labels_dir).glob('*_agbm.tif'))


def split_train_val(dataset, val_fraction=0.2, seed=0):
    """Random split as in the original training (80% / 20% of the BioMassters training chips).
    The original split was not seeded; a fixed seed makes this one repeatable."""
    n_train = int((1 - val_fraction) * len(dataset))
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.random_split(dataset, [n_train, len(dataset) - n_train], generator=generator)
