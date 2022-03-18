import pytorch_lightning as pl
import torch
import typing as th
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from ...model import UNet

@MODEL_REGISTRY
class InpaintingTrainer(pl.LightningModule):
    def __init__(
            self,
            model_params: th.Optional[dict] = None,
            **kwargs,
    ):
        self.save_hyperparameters()
        super().__init__()
        self.model = UNet(**(self.hparams.model_params or dict()))

    def step(self, batch, batch_idx: th.Optional[int] = None, optimizer_idx: th.Optional[int] = None, loop_name: str='train'):
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = ((outputs - targets)**2).sum()
        self.log(f'loss/{loop_name}', loss)
        return loss if loop_name == 'train' else None

    def training_step(self, batch, batch_idx: th.Optional[int] = None, optimizer_idx: th.Optional[int] = None):
        return self.step(batch, batch_idx=batch_idx, optimizer_idx=optimizer_idx, loop_name='train')

    def validation_step(self, batch, batch_idx: th.Optional[int] = None, optimizer_idx: th.Optional[int] = None):
        return self.step(batch, batch_idx=batch_idx, optimizer_idx=optimizer_idx, loop_name='val')
