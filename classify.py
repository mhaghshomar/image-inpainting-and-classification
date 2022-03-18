import torch
from osail.data import ClassificationDataModule
from osail.trainer import ClassificationTrainer
from pytorch_lightning.utilities.cli import LightningCLI


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_optimizer_args((torch.optim.SGD, torch.optim.Adam))
        parser.add_lr_scheduler_args((torch.optim.lr_scheduler.StepLR, torch.optim.lr_scheduler._LRScheduler))


if __name__ == '__main__':
    MyLightningCLI(model_class=ClassificationTrainer,  datamodule_class=ClassificationDataModule)
