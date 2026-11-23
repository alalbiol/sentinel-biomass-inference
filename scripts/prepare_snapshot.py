"""Build the EOTDL model snapshot of the release in VERSION from a clean, tagged source tree."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]
VERSION=(ROOT/'VERSION').read_text().strip()
TAG=f'v{VERSION}'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/f'dist/{TAG}/sentinel-biomass-inference')
    args=p.parse_args()
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise SystemExit('Commit source changes before building a release snapshot.')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    tagged=subprocess.check_output(['git','rev-parse',f'{TAG}^{{}}'],cwd=ROOT,text=True).strip()
    if commit!=tagged:raise SystemExit(f'Check out {TAG} to build this snapshot.')
    sources=[ROOT/p for p in subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines() if p!='.gitignore']
    models=json.loads((ROOT/'provenance/checkpoints.json').read_text())['models']
    for model in models:
        path=ROOT/'checkpoints'/model['filename']
        if digest(path)!=model['sha256']:raise ValueError(f'Checkpoint hash mismatch: {path}')
        sources.append(path)
    args.output.mkdir(parents=True,exist_ok=False)
    entries=[]
    for source in sources:
        if source.is_symlink():raise ValueError(f'Symlink in snapshot: {source}')
        relative=source.relative_to(ROOT);out=args.output/relative
        out.parent.mkdir(parents=True,exist_ok=True)
        # Copy files rather than link, so the snapshot cannot change through the source tree.
        shutil.copyfile(source,out)
        entries.append({'path':relative.as_posix(),'bytes':out.stat().st_size,'sha256':digest(out)})
    manifest={'release':TAG,'github_commit':commit,
              'github_release':f'https://github.com/alalbiol/sentinel-biomass-inference/releases/tag/{TAG}',
              'eotdl_model_name':'sentinel-biomass-inference','eotdl_version':None,
              'files':entries,'note':'Manifest lists every payload file except itself; EOTDL version is assigned on upload.'}
    (args.output/'release-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Snapshot ready: {args.output}; {len(entries)} files; {sum(e["bytes"] for e in entries):,} bytes. Nothing uploaded.')


if __name__=='__main__':main()
