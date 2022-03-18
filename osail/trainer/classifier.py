import pytorch_lightning as pl
import torch
import typing as th
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from ..model import Classifier
from torch.nn import functional as F
from torchmetrics import Accuracy

@MODEL_REGISTRY
class ClassificationTrainer(pl.LightningModule):
    def __init__(
            self,
            model_params: th.Optional[dict] = None,
            **kwargs,
    ):
        self.save_hyperparameters()
        super().__init__()
        self.model = Classifier(**(self.hparams.model_params or dict()))
        self.val_acc = Accuracy(num_classes=2)
        self.train_acc = Accuracy(num_classes=2)
        
    def step(self, batch, batch_idx: th.Optional[int] = None, optimizer_idx: th.Optional[int] = None, loop_name: str='train'):
        inputs, targets = batch
        logits = self.model(inputs)
        loss = F.cross_entropy(logits, targets)
        getattr(self, f'{loop_name}_acc')(torch.argmax(logits), targets)
        self.log(f'acc/{loop_name}', getattr(self, f'{loop_name}_acc'), prog_bar=True)
        self.log(f'loss/{loop_name}', loss)
        return loss if loop_name == 'train' else None

    def training_step(self, batch, batch_idx: th.Optional[int] = None, optimizer_idx: th.Optional[int] = None):
        return self.step(batch, batch_idx=batch_idx, optimizer_idx=optimizer_idx, loop_name='train')

    def validation_step(self, batch, batch_idx: th.Optional[int] = None, optimizer_idx: th.Optional[int] = None):
        return self.step(batch, batch_idx=batch_idx, optimizer_idx=optimizer_idx, loop_name='val')
