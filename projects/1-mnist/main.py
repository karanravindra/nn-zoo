import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from ml_zoo import (
    MNISTDataModule,
    MNISTDataModuleConfig,
    Classifier,
    ClassifierConfig,
    LeNet,
    LeNetConfig,
)

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import wandb
import torch
import torch.nn as nn
import torchvision.transforms as transforms


def main():
    config = MNISTDataModuleConfig(
        data_dir="data",
        batch_size=128,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        transforms=transforms.Compose([transforms.ToTensor(), transforms.Resize((32, 32))]),
        use_qmnist=False,
    )

    dm = MNISTDataModule(config)

    lenetConfig = LeNetConfig(version=5)
    model = LeNet(lenetConfig)

    classifierConfig = ClassifierConfig(
        model=model,
        dm=dm,
        optim="SGD",
        optim_args={
            "lr": 0.01,
            "momentum": 0.9,
            "weight_decay": 0,
        },
    )

    classifier = Classifier(classifierConfig)

    logger = WandbLogger(
        project="mnist",
        dir="projects/1-mnist/logs",
        save_dir="projects/1-mnist/logs",
        log_model=True,
    )

    logger.watch(model, log="all", log_freq=100, log_graph=True)

    trainer = pl.Trainer(
        logger=logger,
        default_root_dir="projects/1-mnist/logs",
        max_epochs=100,
        check_val_every_n_epoch=1,
    )

    trainer.fit(classifier)
    trainer.test(classifier)


if __name__ == "__main__":
    main()
