import sys
import os

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger

import torch.nn as nn
import torchvision.transforms as transforms

from torchinfo import summary

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from ml_zoo import (
    TinyImageNetDataModuleConfig,
    TinyImageNetDataModule,
    Classifier,
    ClassifierConfig,
    ResNet,
    ResNetConfig,
)


def main(model: nn.Module, run_name: str = "tiny-imagenet"):
    config = TinyImageNetDataModuleConfig(
        data_dir="data",
        batch_size=64,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        transforms=transforms.Compose([transforms.ToTensor()]),
    )

    dm = TinyImageNetDataModule(config)

    classifierConfig = ClassifierConfig(
        model=model,
        dm=dm,
        optim="SGD",
        optim_args={
            "lr": 0.1,
            "momentum": 0.9,
            "weight_decay": 1e-4,
        },
        scheduler="ReduceLROnPlateau",
        scheduler_args={
            "factor": 0.1,
            "patience": 1,
            "threshold": 1e-4,
            "min_lr": 1e-6,
        },
        _log_test_table=False,
    )

    classifier = Classifier(classifierConfig)

    logger = WandbLogger(
        name=run_name,
        project="tiny-imagenet",
        dir="projects/3-tiny-imagenet/logs",
        save_dir="projects/3-tiny-imagenet/logs",
        log_model=True,
    )

    logger.watch(model, log="all", log_freq=1, log_graph=True)

    summary(model, input_size=(1, *model.config.sample_size))

    trainer = pl.Trainer(
        logger=logger,
        default_root_dir="projects/3-tiny-imagenet/logs",
        max_epochs=100,
        val_check_interval=0.25,
        enable_model_summary=False,
    )

    trainer.fit(classifier)
    trainer.test(classifier)

    logger.experiment.finish()


if __name__ == "__main__":
    models = {
        "resnet18": ResNet(ResNetConfig(version=18, sample_size=(3, 64, 64), num_classes=200)),
    }

    for name, model in models.items():
        main(model, name)
        break
