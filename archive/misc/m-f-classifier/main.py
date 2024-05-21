import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger

import torch.nn as nn
import torchvision.transforms as transforms

from torchinfo import summary

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from ml_zoo import (
    CelebAHQDataModuleConfig,
    CelebAHQDataModule,
    Classifier,
    ClassifierConfig,
    # ResNet,
    # ResNetConfig,
    LeNet,
    LeNetConfig,
)


def main(model: nn.Module, run_name: str = "classifier"):
    # Create DataModule
    dm_config = CelebAHQDataModuleConfig(
        data_dir="data",
        batch_size=64,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        transforms=transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((32, 32))]
        ),
    )

    dm = CelebAHQDataModule(dm_config)

    # Create Classifier
    classifier_config = ClassifierConfig(
        model=model,
        dm=dm,
        optim="SGD",
        optim_args={"lr": 0.1, "momentum": 0.9},
    )

    classifier = Classifier(classifier_config)

    # Log model
    logger = WandbLogger(
        name=run_name,
        project="mf-classifier",
        save_dir="archive/misc/m-f-classifier/logs",
        log_model=True,
    )

    # Train
    trainer = pl.Trainer(
        logger=logger,
        default_root_dir="archive/misc/m-f-classifier/logs",
        max_steps=1000,
        val_check_interval=100,
        log_every_n_steps=1,
        enable_model_summary=False,
        profiler="simple",
    )

    logger.watch(model, log="all", log_freq=100, log_graph=True)

    summary(model, input_size=(1, 3, 32, 32))

    trainer.fit(classifier)
    # trainer.test(classifier)

    logger.experiment.finish()


if __name__ == "__main__":
    models = {
        # "resnet": ResNet(ResNetConfig(version=None, sample_size=(3, 32, 32), num_classes=2, blocks=(1, 1, 1, 1), block_channels=(64, 64, 64, 64), dropouts=0.5)),
        "lenet": LeNet(LeNetConfig(version=5, input_channels=3, num_classes=2, dropouts=[0.5, 0.5, 0.5], activation=nn.LeakyReLU())),
    }

    for name, model in models.items():
        main(model, name)
