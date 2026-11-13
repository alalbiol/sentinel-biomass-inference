"""Download one 256x256 Sentinel-1/2 patch on a shared 10 m grid.

Authenticate with `earthengine authenticate` first. This is a new acquisition
recipe, not a byte-identical reconstruction of the BioMassters training data.
"""
import argparse
from datetime import date
import io
import json
import math
import os
from pathlib import Path
import sys
import time
from urllib.request import urlopen
import zipfile

import ee
import numpy as np
from pyproj import Transformer
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from biomass.io import S1_BANDS, S2_BANDS


def month_date(start, offset):
    year, index = divmod(start.year * 12 + start.month - 1 + offset, 12)
    return date(year, index + 1, 1).isoformat()


def s1_image(start, end, region, recipe):
    coll = (ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(region).filterDate(start,end)
            .filter(ee.Filter.eq('instrumentMode','IW'))
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VV'))
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VH')))
    parts=[]
    for direction in ['ASCENDING','DESCENDING']:
        subset=coll.filter(ee.Filter.eq('orbitProperties_pass',direction)).select(['VV','VH'])
        if recipe=='biomassters-style':
            subset=subset.map(lambda image: image.updateMask(image.gte(-25)))
            composite=subset.mean()
        else:
            composite=subset.median()
        blank=ee.Image.constant([-9999,-9999]).rename(['VV','VH'])
        parts.append(ee.Image(ee.Algorithms.If(subset.size().gt(0),composite,blank)).unmask(-9999,sameFootprint=False))
    return ee.Image.cat(parts).rename(S1_BANDS).toFloat(), coll.aggregate_array('system:index')


def s2_image(start,end,region):
    optical=ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(region).filterDate(start,end)
    cloud=ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY').filterBounds(region).filterDate(start,end)
    joined=ee.ImageCollection(ee.Join.saveFirst('cloud').apply(optical,cloud,ee.Filter.equals(leftField='system:index',rightField='system:index'))).filter(ee.Filter.notNull(['cloud']))
    def score(image):
        image=ee.Image(image)
        cp=ee.Image(image.get('cloud')).select('probability')
        return image.set({'cloud_mean':cp.reduceRegion(ee.Reducer.mean(),region,20,maxPixels=1e6).get('probability'),
                          'coverage':image.select('B2').mask().reduceRegion(ee.Reducer.mean(),region,20,maxPixels=1e6).get('B2')})
    candidates=joined.map(score).filter(ee.Filter.gte('coverage',0.999)).filter(ee.Filter.notNull(['cloud_mean'])).sort('cloud_mean')
    count=candidates.size().getInfo()
    if count==0:return None,{'available_s2_scenes':0}
    chosen=ee.Image(candidates.first())
    info=chosen.toDictionary(['system:index','cloud_mean']).getInfo()
    cp=ee.Image(chosen.get('cloud')).select('probability').multiply(2.55).rename('CLP')
    image=chosen.select(S2_BANDS[:-1]).addBands(cp).unmask(0,sameFootprint=False).toFloat()
    return image,{'available_s2_scenes':count,'chosen_s2_scene':info['system:index'],'cloud_mean_percent':info['cloud_mean']}


def download_image(image,path,params,count):
    partial=path.with_suffix('.part')
    for attempt in range(3):
        try:
            with urlopen(image.getDownloadURL(params),timeout=180) as response:content=response.read()
            if content[:2]==b'PK':
                with zipfile.ZipFile(io.BytesIO(content)) as z:
                    members=[n for n in z.namelist() if n.lower().endswith('.tif')]
                    if len(members)!=1:raise ValueError('Expected one multi-band TIFF')
                    content=z.read(members[0])
            partial.write_bytes(content)
            with rasterio.open(partial) as src:
                if (src.count,src.width,src.height)!=(count,256,256):raise ValueError('Unexpected downloaded raster dimensions')
                if not np.isfinite(src.read()).all():raise ValueError('Non-finite downloaded values')
            partial.replace(path)
            return
        except Exception:
            partial.unlink(missing_ok=True)
            if attempt==2:raise
            time.sleep(attempt+1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lon',type=float,required=True)
    p.add_argument('--lat',type=float,required=True)
    p.add_argument('--start-date',type=date.fromisoformat,required=True,help='First day of month; e.g. 2020-09-01 for the year ending August 2021')
    p.add_argument('--chip-id',default='custom_patch')
    p.add_argument('--output-dir',type=Path,default=Path('downloads/custom_patch'))
    p.add_argument('--project',default=os.environ.get('EE_PROJECT') or os.environ.get('GEE_PROJECT'))
    p.add_argument('--crs',help='Projected CRS in metres; default: local UTM')
    p.add_argument('--recipe',choices=['biomassters-style','spain-evaluation'],default='biomassters-style')
    p.add_argument('--months',type=int,default=12,choices=range(1,13),help='Use 12 for inference; smaller values are download diagnostics only')
    args=p.parse_args()
    if not args.project:p.error('--project, EE_PROJECT, or GEE_PROJECT is required')
    if not (-180<=args.lon<180 and -80<=args.lat<=84):p.error('Coordinates must be inside UTM coverage')
    if args.start_date.day!=1:p.error('--start-date must be the first day of a month')
    if Path(args.chip_id).name!=args.chip_id or args.chip_id in ('.','..'):p.error('Invalid chip ID')
    zone=int((args.lon+180)//6)+1
    crs=args.crs or f'EPSG:{(32600 if args.lat>=0 else 32700)+zone}'
    transform=Transformer.from_crs('EPSG:4326',crs,always_xy=True)
    if not transform.target_crs.is_projected or any(a.unit_name!='metre' for a in transform.target_crs.axis_info):p.error('--crs must be projected in metres')
    x,y=transform.transform(args.lon,args.lat)
    left,top=math.floor(x/10)*10-1280,math.floor(y/10)*10+1280
    config={'chip_id':args.chip_id,'lon':args.lon,'lat':args.lat,'start_date':args.start_date.isoformat(),
            'months':args.months,'recipe':args.recipe,'crs':crs,'crs_transform':[10,0,left,0,-10,top]}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    receipt_path=args.output_dir/'acquisition.json'
    receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else {'configuration':config,'months':[]}
    if receipt['configuration']!=config:raise ValueError('Output folder contains a different acquisition; choose a new output folder')
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    ee.Initialize(project=args.project)
    region=ee.Geometry.Rectangle([left,top-2560,left+2560,top],proj=crs,geodesic=False)
    params={'crs':crs,'crs_transform':config['crs_transform'],'dimensions':[256,256],'format':'GEO_TIFF','filePerBand':False}
    finished={row['month_index'] for row in receipt['months']}
    for month in range(args.months):
        if month in finished:
            row=next(r for r in receipt['months'] if r['month_index']==month)
            files=[args.output_dir/f'{args.chip_id}_S1_{month:02d}.tif']
            if not row['missing_s2']:files.append(args.output_dir/f'{args.chip_id}_S2_{month:02d}.tif')
            if all(f.is_file() for f in files):continue
        start,end=month_date(args.start_date,month),month_date(args.start_date,month+1)
        s1,ids=s1_image(start,end,region,args.recipe)
        s2,info=s2_image(start,end,region)
        download_image(s1,args.output_dir/f'{args.chip_id}_S1_{month:02d}.tif',params,4)
        if s2 is not None:download_image(s2,args.output_dir/f'{args.chip_id}_S2_{month:02d}.tif',params,11)
        else:(args.output_dir/f'{args.chip_id}_S2_{month:02d}.tif').unlink(missing_ok=True)
        row={'month_index':month,'start':start,'end_exclusive':end,'s1_scenes':ids.getInfo(),'missing_s2':s2 is None,**info}
        receipt['months']=[r for r in receipt['months'] if r['month_index']!=month]+[row]
        receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
        print(f'Month {month:02d}: {start}; optical {"missing" if s2 is None else "downloaded"}',flush=True)
    print(f'Downloaded {args.months} months to {args.output_dir}')


if __name__=='__main__':main()
