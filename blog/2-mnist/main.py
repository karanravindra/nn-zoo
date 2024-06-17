import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger

import torch.nn as nn
import torchvision.transforms as transforms

from torchinfo import summary
import torch
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


def main(model: nn.Module, run_name: str = "qmnist"):
    # Create DataModule
    dm_config = MNISTDataModuleConfig(
        data_dir="data",
        batch_size=128,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        transforms=[
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
        ],
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
        project="qmnist",
        dir="blog/2-mist/logs",
        save_dir="blog/2-mist/logs",
        log_model=True,
    )

    # Train
    trainer = pl.Trainer(
        logger=logger,
        default_root_dir="blog/2-mist/logs",
        max_epochs=100,
        check_val_every_n_epoch=1,
        enable_model_summary=False,
    )

    logger.watch(model, log="all", log_freq=100, log_graph=True)

    summary(model, input_size=(1, 1, 32, 32))

    trainer.fit(classifier)
    trainer.test(classifier)

    logger.experiment.finish()


if __name__ == "__main__":
    models = {
        # "one layer fcn": nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(28 * 28, 10),
        # ),
        # "two layer fcn": nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(28 * 28, 128),
        #     nn.Tanh(),
        #     nn.Linear(128, 10),
        # ),
        # "three layer fcn": nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(28 * 28, 256),
        #     nn.Tanh(),
        #     nn.Linear(256, 128),
        #     nn.Tanh(),
        #     nn.Linear(128, 10),
        # ),
        # "lenet 1": LeNet(LeNetConfig(sample_size=(1, 28, 28), version=1)),
        # "lenet 4": LeNet(LeNetConfig(sample_size=(1, 32, 32), version=4)),
        # "lenet 5": LeNet(LeNetConfig(sample_size=(1, 32, 32), version=5)),
        # "lenet 5 relu": LeNet(LeNetConfig(sample_size=(1, 32, 32), version=5, activation="ReLU")),
        # "lenet 5 relu max-pool": LeNet(LeNetConfig(sample_size=(1, 32, 32), version=5, activation="ReLU", poolings="MaxPool2d")),
        # "lenet 5 relu max-pool drop1": LeNet(LeNetConfig(sample_size=(1, 32, 32), version=5, activation="ReLU", poolings="MaxPool2d", dropouts=[0.5, 0, 0])),
        # "lenet 5 relu max-pool drop2": LeNet(LeNetConfig(sample_size=(1, 32, 32), version=5, activation="ReLU", poolings="MaxPool2d", dropouts=[0.5, 0.5, 0])),
        # "lenet 5 relu max-pool drop3": LeNet(LeNetConfig(sample_size=(1, 32, 32), version=5, activation="ReLU", poolings="MaxPool2d", dropouts=[0.5, 0.5, 0.5])),
    }

    for name, model in models.items():
        main(model, name)
