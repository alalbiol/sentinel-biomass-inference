"""Read the raw, 12-month BioMassters-style raster inputs."""
from pathlib import Path
import numpy as np
import rasterio

S1_BANDS = ['VV_asc', 'VH_asc', 'VV_desc', 'VH_desc']
S2_BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12', 'CLP']


def load_chip(directory, chip_id):
    if not chip_id or Path(chip_id).name != chip_id or chip_id in ('.', '..'):
        raise ValueError('chip_id must be a filename-safe identifier')
    directory = Path(directory)
    frames, missing, reference = [], [], None
    for month in range(12):
        path = directory / f'{chip_id}_S1_{month:02d}.tif'
        with rasterio.open(path) as src:
            if (src.count, src.height, src.width) != (4, 256, 256):
                raise ValueError(f'{path}: expected four bands, 256x256 pixels')
            profile = src.profile.copy()
            grid = (src.crs, src.transform)
            if reference is None:
                reference = (grid, profile)
            if grid != reference[0]:
                raise ValueError(f'{path}: raster grids differ')
            s1 = src.read().astype('float32')
        s1[s1 == -9999] = 0  # identical to the original norm=False loader
        optical = directory / f'{chip_id}_S2_{month:02d}.tif'
        if optical.exists():
            with rasterio.open(optical) as src:
                if (src.count, src.height, src.width) != (11, 256, 256):
                    raise ValueError(f'{optical}: expected eleven bands, 256x256 pixels')
                if (src.crs, src.transform) != reference[0]:
                    raise ValueError(f'{optical}: raster grids differ')
                s2 = src.read().astype('float32')
            missing.append(False)
        else:
            s2 = np.zeros((11, 256, 256), dtype='float32')
            missing.append(True)
        frame = np.concatenate([s1, s2])
        if not np.isfinite(frame).all():
            raise ValueError(f'{chip_id}, month {month}: non-finite input values')
        frames.append(frame)
    if all(missing):
        raise ValueError('At least one Sentinel-2 month is required')
    return np.stack(frames), np.asarray(missing, dtype=bool), reference[1]


def save_prediction(path, prediction, profile):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(profile, count=1, dtype='float32', nodata=None, compress='deflate')
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(prediction.astype('float32'), 1)
        dst.set_band_description(1, 'aboveground_biomass_Mg_ha')
        dst.update_tags(units='Mg/ha', quantity='aboveground dry biomass', clipping='none')
