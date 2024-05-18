import sys
import os

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger

import torch.nn as nn
import torchvision.transforms as transforms

from torchinfo import summary

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from ml_zoo import (
    MNISTDataModule,
    MNISTDataModuleConfig,
    Classifier,
    ClassifierConfig,
    LeNet,
    LeNetConfig,
)


def main(model: nn.Module, run_name: str = "mnist"):
    # Create DataModule
    dm_config = MNISTDataModuleConfig(
        data_dir="data",
        batch_size=128,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        transforms=transforms.Compose(
            [transforms.ToTensor()]  # , transforms.Resize((32, 32))]
        ),
        use_qmnist=True,
    )

    dm = MNISTDataModule(dm_config)

    # Create Classifier
    classifier_config = ClassifierConfig(
        model=model,
        dm=dm,
        optim="SGD",
        optim_args={"lr": 0.01, "momentum": 0.9},
    )

    classifier = Classifier(classifier_config)

    # Log model
    logger = WandbLogger(
        name=run_name,
        project="test",
        dir="projects/1-mnist/logs",
        save_dir="projects/1-mnist/logs",
        log_model=True,
    )

    # Train
    trainer = pl.Trainer(
        logger=logger,
        default_root_dir="projects/1-mnist/logs",
        max_epochs=50,
        check_val_every_n_epoch=1,
        enable_model_summary=False,
    )

    logger.watch(model, log="all", log_freq=100, log_graph=True)

    summary(model, input_size=(1, 1, 28, 28))

    trainer.fit(classifier)
    trainer.test(classifier)

    logger.experiment.finish()


if __name__ == "__main__":
    models = {
        "one layer fcn": nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 10),
        ),
        "two layer fcn": nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.Tanh(),
            nn.Linear(128, 10),
        ),
        "three layer fcn": nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 10),
        ),
        "lenet 1": LeNet(LeNetConfig(version=1)),
        # "lenet 4": LeNet(LeNetConfig(version=4)),
        # "lenet 5 small lr": LeNet(LeNetConfig(version=5)),
    }

    for name, model in models.items():
        main(model, name)
