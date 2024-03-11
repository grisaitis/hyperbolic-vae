import pytorch_lightning as pl
import geoopt

"""
action plan
- get something working
- figure out scvi architecture... what was the net architecture in terms of layers, activations, etc.?
"""


class VAEHyperbolicRNASeq(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float = 1e-3,
        latent_manifold: geoopt.Stereographic = geoopt.PoincareBall(c=1.0),
        hidden_dimensions: int = 16,
    ):
        super().__init__()
        # set hyperparameters (learning rate, hidden dimensions, dropout, etc.)
        # set pytorch module (layers, networks, etc.)
        # set likelihood (negative binomial)
        # set

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass

    def train_dataloader(self):
        pass