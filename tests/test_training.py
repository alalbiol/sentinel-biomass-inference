"""Training-code checks that need neither BioMassters nor a GPU (python -m unittest discover tests)."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import rasterio
import torch
from rasterio.transform import from_origin
from safetensors.torch import load_file

from biomass.inference import MODEL_IDS, load_model
from biomass.io import load_chip
from training.convert_biomassters import convert
from training.data import BioMassters, read_chip_tifs
from training.export_checkpoint import export
from training.module import BiomassModule, build_model

ROOT = Path(__file__).resolve().parents[1]


class FakeChip(unittest.TestCase):
    """One BioMassters-style chip with S1 no-data and a missing Sentinel-2 month."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        rng = np.random.default_rng(0)
        profile = dict(driver='GTiff', height=256, width=256, dtype='float32', crs='EPSG:3035',
                       transform=from_origin(0, 2560, 10, 10))
        for month in range(12):
            s1 = rng.normal(-12, 4, (4, 256, 256)).astype('float32')
            s1[1, :8] = -9999
            self.write(f'chip_S1_{month:02d}.tif', s1, profile)
            if month != 3:
                self.write(f'chip_S2_{month:02d}.tif', rng.uniform(0, 5000, (11, 256, 256)).astype('float32'), profile)
        self.write('chip_agbm.tif', rng.uniform(0, 300, (1, 256, 256)).astype('float32'), profile)

    def write(self, name, data, profile):
        with rasterio.open(self.root / name, 'w', **dict(profile, count=len(data))) as dst:
            dst.write(data)


class DataTests(FakeChip):
    def test_training_reader_matches_inference_reader(self):
        features, missing = read_chip_tifs('chip', self.root)
        images, mask, _ = load_chip(self.root, 'chip')
        np.testing.assert_array_equal(features, images)
        self.assertEqual(missing.tolist(), [3])
        self.assertEqual(np.flatnonzero(mask).tolist(), [3])
        self.assertTrue((features[:, 1, :8] == 0).all())

    def test_safetensors_and_tif_inputs_are_identical(self):
        out = self.root / 'converted'
        out.mkdir()
        convert('chip', self.root, out)
        a = BioMassters(['chip'], out, self.root, 'safetensors')[0]
        b = BioMassters(['chip'], self.root, self.root, 'tif')[0]
        for x, y in zip(a, b):
            self.assertTrue(torch.equal(x, y))
        self.assertEqual(a[0].shape, (12, 15, 256, 256))
        self.assertEqual(a[1].nonzero().flatten().tolist(), [3])


class ModelTests(FakeChip):
    def test_one_training_step_per_fusion_mode(self):
        x, mask, y = BioMassters(['chip'], self.root, self.root, 'tif')[0]
        for model_id in ('mobilenetv4_early', 'mobilenetv4_late'):
            module = BiomassModule(model_id, pretrained=False)
            opt = torch.optim.AdamW(module.parameters(), lr=1e-3, weight_decay=0.01)
            module.train()
            pred = module(x[None], mask[None].clone()).squeeze(1)
            loss = torch.nn.functional.mse_loss(pred, y[None])
            loss.backward()
            opt.step()
            self.assertTrue(torch.isfinite(loss))

    def test_export_round_trip_loads_strictly(self):
        module = BiomassModule('mobilenetv4_late', pretrained=False)
        ckpt = self.root / 'm.ckpt'
        torch.save({'state_dict': module.state_dict()}, ckpt)
        target = export(ckpt, self.root / 'mobilenetv4_late.safetensors', 'mobilenetv4_late')
        model = load_model('mobilenetv4_late', self.root, 'cpu')
        for k, v in load_file(str(target)).items():
            self.assertTrue(torch.equal(v, model.state_dict()[k]))

    @unittest.skipUnless((ROOT / 'checkpoints').is_dir(), 'released checkpoints not downloaded')
    def test_released_checkpoints_fit_the_training_architecture(self):
        for model_id in MODEL_IDS:
            state = load_file(str(ROOT / 'checkpoints' / f'{model_id}.safetensors'))
            build_model(model_id, pretrained=False).load_state_dict(state, strict=True)


if __name__ == '__main__':
    unittest.main()
