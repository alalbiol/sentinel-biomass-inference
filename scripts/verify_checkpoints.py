"""Check checkpoint hashes after downloading an EOTDL model snapshot."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',type=Path,default=ROOT/'checkpoints')
    args=p.parse_args()
    for entry in json.loads((ROOT/'provenance/checkpoints.json').read_text())['models']:
        path=args.model_dir/entry['filename'];h=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
        if path.stat().st_size!=entry['bytes'] or h.hexdigest()!=entry['sha256']:
            raise ValueError(f'Checkpoint hash mismatch: {path}')
        print(f'{entry["model_id"]}: verified')


if __name__=='__main__':main()
