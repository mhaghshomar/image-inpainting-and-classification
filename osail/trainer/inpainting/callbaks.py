import pytorch_lightning as pl
import torchvision


class VisualizeStep(pl.Callback):
    """Log the reconstruction results of a step
    Attributes:
        every_n_steps (int): log interval (default: 50)
        log_train (bool): whether to log training steps
        log_val (bool): whether to log validation steps
    """

    def __init__(self, every_n_steps: int = 50, log_train: bool = True, log_val: bool = False):
        super().__init__()
        self.log_train, self.log_val = log_train, log_val
        self.every_n_steps = every_n_steps

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch,
        batch_idx: int,
        unused = 0,
    ) -> None:
        """Called when the train batch ends."""
        if self.log_train:
            self.on_step_end(trainer=trainer, pl_module=pl_module, batch=batch, batch_idx=batch_idx, loop_name='train')
    
    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the validation batch ends."""
        if self.log_val:
            self.on_step_end(trainer=trainer, pl_module=pl_module, batch=batch, batch_idx=batch_idx, loop_name='val')

    def on_step_end(self, trainer, pl_module, batch, batch_idx, loop_name:str="train"):
        if self.every_n_steps and batch_idx % self.every_n_steps == 0:
            inputs, targets = batch
            trainer.logger.experiment.add_image(f"{loop_name}/inputs", torchvision.utils.make_grid(inputs, nrow=4, normalize=True, range=(0, 1)), global_step=trainer.global_step)
            trainer.logger.experiment.add_image(f"{loop_name}/outputs", torchvision.utils.make_grid(pl_module.model(inputs), nrow=4, normalize=True, range=(0, 1)), global_step=trainer.global_step)
            trainer.logger.experiment.add_image(f"{loop_name}/targets", torchvision.utils.make_grid(targets, nrow=4, normalize=True, range=(0, 1)), global_step=trainer.global_step)
              

