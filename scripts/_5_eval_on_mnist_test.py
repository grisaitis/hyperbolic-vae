import pathlib

from _5_train_vae_hyperbolic_mnist import trainer, vae_experiment

from hyperbolic_vae.data.mnist_v2 import mnist_data_module

CHECKPOINT_PATHS = pathlib.Path("/home/jupyter/hyperbolic-vae/checkpoints/mnist/lightning_logs")
paths = CHECKPOINT_PATHS.glob("version_*/checkpoints/*.ckpt")

for checkpoint_path in paths:
    trainer.test(
        model=vae_experiment,
        ckpt_path=checkpoint_path,
        datamodule=mnist_data_module,
    )

# trainer_euclidean.test(ckpt_path="best", datamodule=mnist_data_module)
# trainer_hyperbolic.test(ckpt_path="best", datamodule=mnist_data_module)
