import pytorch_lightning as pl
import torch
import torchvision


class GenerateCallback(pl.Callback):
    def __init__(self, input_imgs: torch.Tensor, every_n_epochs: int = 1):
        super().__init__()
        self.input_imgs = input_imgs
        self.every_n_epochs = every_n_epochs

    @classmethod
    def from_data_module(
        cls,
        data_module: pl.LightningDataModule,
        n_images: int = 8,
        every_n_epochs: int = 1,
    ):
        dataset_train = data_module.train_dataloader().dataset
        images = torch.stack([dataset_train[i][0] for i in range(n_images)], dim=0)
        return cls(images, every_n_epochs=every_n_epochs)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            input_imgs = self.input_imgs.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                # reconst_imgs = pl_module(input_imgs)
                mu, log_var, z, x_hat = pl_module(input_imgs)
                reconst_imgs = x_hat
                pl_module.train()
            # Plot and add to tensorboard
            imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0, 1)
            grid = torchvision.utils.make_grid(imgs, nrow=2, normalize=True)
            trainer.logger.experiment.add_image("Reconstructions", grid, global_step=trainer.global_step)
