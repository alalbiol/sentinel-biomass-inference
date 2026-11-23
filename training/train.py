"""Train one of the ten released models on the BioMassters training split.

    python -m training.train --model mobilenetv4_early \
        --features-dir data/biomassters/train_safetensors --labels-dir data/biomassters/train_agbm

The defaults are the settings of the released checkpoints, read from their Lightning training
state (provenance/training.json): 20 epochs, batch size 3 on one GPU, AdamW (learning rate 1e-3,
weight decay 0.01), cosine annealing to 1e-6 over the 20 epochs, MSE loss, ImageNet-initialized
timm encoder, 80% of the 8,689 training chips for fitting and 20% for validation, and the
checkpoint of the last epoch. Writes <output-dir>/<model>/ with the Lightning checkpoint, CSV logs
and, at the end, the inference-only <model>.safetensors.
"""
import argparse
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from biomass.inference import MODEL_IDS
from training.data import BioMassters, list_chips, split_train_val
from training.export_checkpoint import export
from training.module import BiomassModule


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--model', choices=MODEL_IDS, required=True)
    p.add_argument('--features-dir', type=Path, required=True, help='converted .safetensors (or the GeoTIFFs with --format tif)')
    p.add_argument('--labels-dir', type=Path, required=True, help='<chip_id>_agbm.tif labels')
    p.add_argument('--metafile', type=Path, help='CSV with a chip_id column (default: every label in --labels-dir)')
    p.add_argument('--format', choices=['safetensors', 'tif'], default='safetensors')
    p.add_argument('--output-dir', type=Path, default=Path('runs'))
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=3)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--eta-min', type=float, default=1e-6)
    p.add_argument('--val-fraction', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=0, help='seed for the train/validation split, decoder initialization and batch order')
    p.add_argument('--no-pretrained', action='store_true', help='random encoder initialization instead of ImageNet')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--accelerator', default='auto')
    p.add_argument('--precision', default='32-true',
                   help='32-true as in the original runs; bf16-mixed fits EfficientNet-B5/V2-M in 12 GB of GPU memory')
    p.add_argument('--max-steps', type=int, default=-1, help='stop early (smoke tests)')
    p.add_argument('--limit-val-batches', type=float, default=1.0)
    args = p.parse_args()

    L.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision('medium')  # TF32 matmuls, as in the original runs
    chips = list_chips(args.labels_dir, args.metafile)
    train_set, val_set = split_train_val(BioMassters(chips, args.features_dir, args.labels_dir, args.format),
                                         args.val_fraction, args.seed)
    loader = lambda ds, shuffle: torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                                                             num_workers=args.workers, persistent_workers=args.workers > 0)
    print(f'{len(chips)} chips: {len(train_set)} for fitting, {len(val_set)} for validation', flush=True)

    out = args.output_dir / args.model
    module = BiomassModule(args.model, pretrained=not args.no_pretrained, lr=args.lr,
                           weight_decay=args.weight_decay, eta_min=args.eta_min)
    checkpoint = ModelCheckpoint(dirpath=out, filename=args.model, every_n_epochs=1)  # keeps the last epoch
    trainer = L.Trainer(max_epochs=args.epochs, max_steps=args.max_steps, accelerator=args.accelerator, devices=1,
                        precision=args.precision,
                        logger=CSVLogger(out, name='logs'), callbacks=[checkpoint, LearningRateMonitor()],
                        limit_val_batches=args.limit_val_batches, enable_progress_bar=True)
    trainer.fit(module, loader(train_set, True), loader(val_set, False))
    ckpt = Path(checkpoint.last_model_path or checkpoint.best_model_path)
    target = export(ckpt, out / f'{args.model}.safetensors', args.model)
    print(f'Checkpoint: {ckpt}\nInference weights: {target}', flush=True)


if __name__ == '__main__':
    main()
