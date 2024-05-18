import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from ml_zoo import (
    CIFARDataModule,
    CIFARDataModuleConfig,
    Classifier,
    ClassifierConfig,
    LeNet,
    LeNetConfig,
)

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger

import torch.nn as nn
import torchvision.transforms as transforms


def main(model: nn.Module, project_name: str = "cifar"):
    config = CIFARDataModuleConfig(
        data_dir="data",
        batch_size=128,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        transforms=transforms.Compose([transforms.ToTensor()]),
        use_cifar100=False,
    )

    dm = CIFARDataModule(config)

    classifierConfig = ClassifierConfig(
        model=model,
        dm=dm,
        optim="SGD",
        optim_args={
            "lr": 0.02,
            "momentum": 0.9,
            "weight_decay": 0,
        },
        _log_test_table=False,
    )

    classifier = Classifier(classifierConfig)

    logger = WandbLogger(
        name=project_name,
        project="cifar",
        dir="projects/2-cifar/logs",
        save_dir="projects/2-cifar/logs",
        log_model=True,
    )

    logger.watch(model, log="all", log_freq=100, log_graph=True)

    trainer = pl.Trainer(
        logger=logger,
        default_root_dir="projects/2-cifar/logs",
        max_epochs=100,
        check_val_every_n_epoch=1,
    )

    trainer.fit(classifier)
    trainer.test(classifier)


if __name__ == "__main__":
    models = {
        "one layer fcn": nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 10),
        ),
        "two layer fcn": nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        ),
        "three layer fcn": nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        ),
        "lenet 1": LeNet(LeNetConfig(version=1)),
        "lenet 4": LeNet(
            LeNetConfig(
                version=4,
                paddings=[2, 0, 0],
            )
        ),
        "lenet 5": LeNet(
            LeNetConfig(
                version=5,
                paddings=[2, 0, 0],
            )
        ),
    }

    for name, model in models.items():
        main(model, name)
