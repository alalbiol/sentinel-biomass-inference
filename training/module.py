"""Lightning module used to train the released models (port of the original AI4DIS CustomModel).

The trained parameters live under `model.` (the UnetVFLOW of biomass/model.py), so a checkpoint
written by this module exports to the released safetensors format with
training/export_checkpoint.py, and the original AI4DIS checkpoints load into it unchanged.
"""
import lightning as L
import torch
from torch import nn

from biomass.inference import MODEL_IDS, configure
from biomass.model import UnetVFLOW


def build_model(model_id, pretrained):
    """UnetVFLOW for a released model id; pretrained=True starts the encoder from timm ImageNet weights."""
    if model_id not in MODEL_IDS:
        raise ValueError(f'Unknown model: {model_id}')
    backbone, mode = model_id.rsplit('_', 1)
    cfg = configure(backbone)
    cfg.pretrained = pretrained
    return UnetVFLOW(cfg, mode=mode)


class BiomassModule(L.LightningModule):
    """Pixel-wise MSE regression of biomass (Mg/ha); AdamW with a cosine schedule over all epochs."""

    def __init__(self, model_id, pretrained=True, lr=1e-3, weight_decay=0.01, eta_min=1e-6):
        super().__init__()
        self.save_hyperparameters()
        self.model = build_model(model_id, pretrained)
        self.criterion = nn.MSELoss()

    def forward(self, x, missing):
        return self.model(x, missing)

    def _step(self, batch, name):
        images, missing, targets = batch
        loss = self.criterion(self.model(images, missing).squeeze(1), targets)
        self.log(f'{name}_rmse', torch.sqrt(loss), on_epoch=True, on_step=False, prog_bar=True,
                 batch_size=images.shape[0])
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        self._step(batch, 'val')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs,
                                                               eta_min=self.hparams.eta_min)
        return [optimizer], [scheduler]
