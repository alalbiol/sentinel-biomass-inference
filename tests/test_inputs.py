"""Input-contract checks independent of model weights and Earth Engine access."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import rasterio
from rasterio.transform import from_origin
from biomass.io import load_chip


class InputTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.profile=dict(driver='GTiff',height=256,width=256,dtype='float32',crs='EPSG:3035',transform=from_origin(0,2560,10,10),compress='deflate')
        for month in range(12):
            self.write(f'test_S1_{month:02d}.tif',np.stack([np.full((256,256),-9999 if band==0 else band,dtype='float32') for band in range(4)]))
            self.write(f'test_S2_{month:02d}.tif',np.stack([np.full((256,256),1000+band,dtype='float32') for band in range(11)]))

    def write(self,name,data,**updates):
        with rasterio.open(self.root/name,'w',**dict(self.profile,count=len(data),**updates)) as dst:dst.write(data)

    def test_band_order_raw_scale_and_s1_nodata(self):
        images,missing,_=load_chip(self.root,'test')
        self.assertEqual(images.shape,(12,15,256,256))
        self.assertFalse(missing.any())
        np.testing.assert_array_equal(images[0,:,0,0],np.r_[0,1,2,3,np.arange(1000,1011)])

    def test_missing_optical_month_retains_radar(self):
        (self.root/'test_S2_03.tif').unlink()
        images,missing,_=load_chip(self.root,'test')
        self.assertEqual(np.flatnonzero(missing).tolist(),[3])
        self.assertEqual(images[3,1,0,0],1)
        self.assertEqual(float(images[3,4:].sum()),0)

    def test_misaligned_grid_is_rejected(self):
        self.write('test_S1_05.tif',np.ones((4,256,256),dtype='float32'),transform=from_origin(10,2560,10,10))
        with self.assertRaisesRegex(ValueError,'grids differ'):load_chip(self.root,'test')

    def test_nonfinite_input_is_rejected(self):
        self.write('test_S2_00.tif',np.full((11,256,256),np.nan,dtype='float32'))
        with self.assertRaisesRegex(ValueError,'non-finite'):load_chip(self.root,'test')

    def test_all_optical_missing_is_rejected(self):
        for p in self.root.glob('*_S2_*.tif'):p.unlink()
        with self.assertRaisesRegex(ValueError,'At least one'):load_chip(self.root,'test')


if __name__=='__main__':unittest.main()
