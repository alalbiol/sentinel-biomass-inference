"""Load inference-only safetensors with strict state matching and run a patch."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from safetensors.torch import load_file
from .io import load_chip, save_prediction
from .model import UnetVFLOW

BACKBONES = ['convnextv2', 'efficientnet', 'efficientnetv2', 'mobilenetv4', 'resnetrs101']
MODEL_IDS = [f'{b}_{m}' for b in BACKBONES for m in ('early', 'late')]


def configure(backbone):
    if backbone not in BACKBONES:
        raise ValueError(backbone)
    if backbone == 'convnextv2':
        indices, channels = [0, 1, 2, 3], [256, 128, 64, 32]
    elif backbone in ('efficientnet', 'efficientnetv2', 'resnetrs101'):
        indices, channels = [0, 1, 2, 3, 4], [256, 160, 128, 64, 32]
    else:
        indices, channels = [0, 1, 2, 3, 4], [256, 128, 64, 32, 16]
    return SimpleNamespace(backbone=backbone, in_channels=15, n_classes=1,
                           pretrained=False, out_indices=indices, dec_channels=channels,
                           dec_attn_type=None)


def load_model(model_id, model_dir, device):
    if model_id not in MODEL_IDS:
        raise ValueError(f'Unknown model: {model_id}')
    backbone, mode = model_id.rsplit('_', 1)
    model = UnetVFLOW(configure(backbone), mode=mode)
    weights = Path(model_dir) / f'{model_id}.safetensors'
    model.load_state_dict(load_file(str(weights)), strict=True)
    return model.eval().to(device)


def predict(model, images, missing, device):
    with torch.inference_mode():
        out = model(torch.from_numpy(images[None]).to(device),
                    torch.from_numpy(missing[None]).to(device))
    result = out[0, 0].cpu().numpy().astype('float32')
    if result.shape != (256, 256) or not np.isfinite(result).all():
        raise ValueError('Model produced an invalid biomass map')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODEL_IDS, default='mobilenetv4_early')
    parser.add_argument('--model-dir', type=Path, default=Path('checkpoints'))
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--chip-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    torch.set_num_threads(min(8, torch.get_num_threads()))
    torch.set_float32_matmul_precision('medium')
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else 'cpu' if args.device == 'auto' else args.device
    images, missing, profile = load_chip(args.input_dir, args.chip_id)
    model = load_model(args.model, args.model_dir, device)
    prediction = predict(model, images, missing, device)
    save_prediction(args.output, prediction, profile)
    record = {'model': args.model, 'chip_id': args.chip_id, 'device': device,
              'missing_optical_months': np.flatnonzero(missing).tolist(),
              'mean_Mg_ha': float(prediction.mean()), 'output': str(args.output),
              'weights_sha256': hashlib.sha256((args.model_dir / f'{args.model}.safetensors').read_bytes()).hexdigest()}
    args.output.with_suffix('.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
