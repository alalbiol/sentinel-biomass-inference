"""Score released or newly trained models on the BioMassters test set (2,773 chips, native 10 m).

    python -m training.evaluate --models mobilenetv4_early \
        --features-dir data/biomassters/test_safetensors --labels-dir data/biomassters/test_agbm
    python -m training.evaluate --ensemble early --features-dir ... --labels-dir ...

Metrics are pooled over every test pixel, as in the original AI4DIS evaluation: RMSE, MAE, R²,
the share of pixels over-estimated (prediction > 2 x max(label, 10 Mg/ha)) or under-estimated
(label >= 10 Mg/ha and prediction < label / 2), and a spatial consistency index (SSIM with a
5 x 5 window and data range 500 Mg/ha, averaged over chips). --ensemble averages the five
encoders of one fusion mode and also reports the mean member standard deviation divided by the
mean prediction. expected/biomassters_test.json holds the original results (Table I).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity

from biomass.inference import MODEL_IDS, load_model
from training.data import BioMassters, list_chips

EPS, OVER, UNDER = 10.0, 2.0, 0.5
SSIM_WIN, SSIM_RANGE = 5, 500.0
# Member order of the original ensemble evaluation (it fixes the floating-point summation order).
ENSEMBLE = ['efficientnet', 'convnextv2', 'efficientnetv2', 'mobilenetv4', 'resnetrs101']


class Accumulator:
    def __init__(self):
        self.sse = self.sae = self.sy = self.syy = 0.0
        self.n = self.over = self.under = 0
        self.ssim, self.std, self.mean_pred = [], 0.0, 0.0

    def add(self, pred, target, std=None):
        d = pred - target
        self.sse += (d ** 2).sum().item()
        self.sae += d.abs().sum().item()
        self.sy += target.sum().item()
        self.syy += (target ** 2).sum().item()
        self.n += target.numel()
        self.over += (pred > OVER * torch.clamp(target, min=EPS)).sum().item()
        self.under += ((target >= EPS) & (pred < UNDER * target)).sum().item()
        for p, t in zip(pred.cpu().numpy(), target.cpu().numpy()):
            self.ssim.append(structural_similarity(t, p, data_range=SSIM_RANGE, win_size=SSIM_WIN))
        if std is not None:
            self.std += std.sum().item()
            self.mean_pred += pred.sum().item()

    def result(self):
        mean = self.sy / self.n
        out = {'n_chips': len(self.ssim), 'n_pixels': self.n, 'rmse': (self.sse / self.n) ** 0.5,
               'mae': self.sae / self.n, 'r2': 1 - self.sse / (self.syy - self.n * mean ** 2),
               'frac_over_pct': 100 * self.over / self.n, 'frac_under_pct': 100 * self.under / self.n,
               'sci': float(np.mean(self.ssim))}
        if self.mean_pred:
            out['uncertainty_ratio'] = self.std / self.mean_pred
        return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--models', nargs='+', choices=MODEL_IDS)
    group.add_argument('--ensemble', choices=['early', 'late'])
    p.add_argument('--model-dir', type=Path, default=Path('checkpoints'), help='folder with <model>.safetensors')
    p.add_argument('--features-dir', type=Path, required=True)
    p.add_argument('--labels-dir', type=Path, required=True)
    p.add_argument('--metafile', type=Path)
    p.add_argument('--format', choices=['safetensors', 'tif'], default='safetensors')
    p.add_argument('--limit', type=int, default=0, help='first N chips only (smoke tests)')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--output', type=Path, help='write the metrics as JSON')
    args = p.parse_args()

    torch.set_float32_matmul_precision('medium')
    chips = list_chips(args.labels_dir, args.metafile)
    if args.limit:
        chips = chips[:args.limit]
    loader = torch.utils.data.DataLoader(BioMassters(chips, args.features_dir, args.labels_dir, args.format),
                                         batch_size=1, num_workers=args.workers)
    results = {}
    if args.ensemble:
        members = [load_model(f'{b}_{args.ensemble}', args.model_dir, args.device) for b in ENSEMBLE]
        acc = Accumulator()
        with torch.inference_mode():
            for images, missing, targets in loader:
                images, missing, targets = images.to(args.device), missing.to(args.device), targets.to(args.device)
                preds = torch.stack([m(images, missing.clone()).squeeze(1) for m in members])
                acc.add(preds.mean(0), targets, std=preds.std(0))
        results[f'ensemble_{args.ensemble}'] = acc.result()
    else:
        for model_id in args.models:
            model, acc = load_model(model_id, args.model_dir, args.device), Accumulator()
            with torch.inference_mode():
                for images, missing, targets in loader:
                    targets = targets.to(args.device)
                    acc.add(model(images.to(args.device), missing.to(args.device)).squeeze(1), targets)
            results[model_id] = acc.result()
            del model
    for name, r in results.items():
        print(f"{name}: RMSE {r['rmse']:.4f}  MAE {r['mae']:.4f}  R2 {r['r2']:.4f}  SCI {r['sci']:.4f}  "
              f"over {r['frac_over_pct']:.4f}%  under {r['frac_under_pct']:.4f}%  ({r['n_chips']} chips)", flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
