"""Compare released inference weights with stored project predictions."""
import argparse
import gc
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from biomass.inference import MODEL_IDS, load_model, predict
from biomass.io import load_chip, save_prediction


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path, default=ROOT/'checkpoints')
    p.add_argument('--models', nargs='+', choices=MODEL_IDS, default=['mobilenetv4_early'])
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', choices=['cuda','cpu'])
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('medium')
    metadata=json.loads((ROOT/'samples/metadata.json').read_text())
    inputs={r['chip_id']:load_chip(ROOT/'samples/inputs',r['chip_id']) for r in metadata}
    rows=[]
    atol=0.25 if args.device=='cpu' else 0.1
    for name in args.models:
        model=load_model(name,args.model_dir,args.device)
        for cid,(images,missing,profile) in inputs.items():
            result=predict(model,images,missing,args.device)
            expected=np.load(ROOT/'samples/expected'/name/f'{cid}.npy')
            np.testing.assert_allclose(result,expected,rtol=1e-3,atol=atol)
            rows.append({'model':name,'chip_id':cid,'max_absolute_difference_Mg_ha':float(np.abs(result-expected).max()),'mean_Mg_ha':float(result.mean()),'status':'passed'})
            print(rows[-1],flush=True)
        del model
        gc.collect()
        if args.device=='cuda':torch.cuda.empty_cache()
    out=ROOT/'outputs';out.mkdir(exist_ok=True)
    (out/'sample-verification.json').write_text(json.dumps({'device':args.device,'rtol':1e-3,'atol_Mg_ha':atol,'comparisons':rows},indent=2)+'\n')


if __name__=='__main__':main()
