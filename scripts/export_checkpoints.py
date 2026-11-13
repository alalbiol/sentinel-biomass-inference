"""Maintainer tool: export the ten original Lightning checkpoints to safetensors."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from biomass.inference import MODEL_IDS


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'checkpoints')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for name in MODEL_IDS:
        source = args.source / (name + '.ckpt')
        target = args.output / (name + '.safetensors')
        raw = torch.load(source, map_location='cpu', weights_only=True)
        assert all(k.startswith('model.') for k in raw['state_dict'])
        state = {k[6:]: v.contiguous() for k, v in raw['state_dict'].items()}
        save_file(state, str(target), metadata={'model_id': name, 'project': 'AI4DIS', 'version': '1.0.0'})
        records.append({'model_id': name, 'source_filename': source.name,
                        'source_sha256': sha256(source), 'filename': target.name,
                        'sha256': sha256(target), 'bytes': target.stat().st_size})
        del raw, state
        print(f'Exported {name}', flush=True)
    manifest = {'release': 'v1.0.0', 'format': 'safetensors', 'models': records}
    (ROOT / 'provenance/checkpoints.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
