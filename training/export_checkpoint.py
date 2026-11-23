"""Export a training checkpoint to the inference-only safetensors format of checkpoints/.

    python -m training.export_checkpoint runs/mobilenetv4_early/mobilenetv4_early.ckpt \
        --model mobilenetv4_early --output checkpoints/mobilenetv4_early.safetensors

Keeps the network tensors unchanged (the `model.` prefix is removed); optimizer and trainer
state are dropped. The same conversion produced the released files (scripts/export_checkpoints.py).
"""
import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file

from biomass.inference import MODEL_IDS


def export(ckpt, target, model_id):
    raw = torch.load(ckpt, map_location='cpu', weights_only=True)
    assert all(k.startswith('model.') for k in raw['state_dict']), 'not a BiomassModule checkpoint'
    state = {k[len('model.'):]: v.contiguous() for k, v in raw['state_dict'].items()}
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(target), metadata={'model_id': model_id, 'project': 'AI4DIS'})
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--model', choices=MODEL_IDS, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    print(export(args.checkpoint, args.output, args.model))


if __name__ == '__main__':
    main()
