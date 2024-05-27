import sys
import os

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger

import torch.nn as nn
import torchvision.transforms as transforms

from torchinfo import summary

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from ml_zoo import (
    CIFARDataModule,
    CIFARDataModuleConfig,
    Classifier,
    ClassifierConfig,
    VGG,
    VGGConfig,
    ResNet,
    ResNetConfig,
)


def main(model: nn.Module, run_name: str = "cifar"):
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
            "lr": 0.01,
            "momentum": 0.9,
            "weight_decay": 0,
        },
        _log_test_table=False,
    )

    classifier = Classifier(classifierConfig)

    logger = WandbLogger(
        name=run_name,
        project="cifar",
        dir="projects/2-cifar/logs",
        save_dir="projects/2-cifar/logs",
        log_model=True,
    )

    logger.watch(model, log="all", log_freq=100, log_graph=True)

    summary(model, input_size=(1, 3, 32, 32))

    trainer = pl.Trainer(
        logger=logger,
        default_root_dir="projects/2-cifar/logs",
        max_epochs=100,
        check_val_every_n_epoch=1,
    )

    trainer.fit(classifier)
    trainer.test(classifier)

    logger.experiment.finish()


if __name__ == "__main__":
    models = {
        "vgg11": VGG(
            VGGConfig(
                version=11,
                num_classes=10,
                sample_size=(3, 32, 32),
                global_pooling_dim=1,
            )
        ),
        "vgg13": VGG(
            VGGConfig(
                version=13,
                num_classes=10,
                sample_size=(3, 32, 32),
                global_pooling_dim=1,
            )
        ),
        "vgg16": VGG(
            VGGConfig(
                version=16,
                num_classes=10,
                sample_size=(3, 32, 32),
                global_pooling_dim=1,
            )
        ),
        "vgg19": VGG(
            VGGConfig(
                version=19,
                num_classes=10,
                sample_size=(3, 32, 32),
                global_pooling_dim=1,
            )
        ),
        "resnet18": ResNet(
            ResNetConfig(
                version=18,
                num_classes=10,
                sample_size=(3, 32, 32),
            )
        ),
        "resnet34": ResNet(
            ResNetConfig(
                version=34,
                num_classes=10,
                sample_size=(3, 32, 32),
            )
        ),
        "resnet50": ResNet(
            ResNetConfig(
                version=50,
                num_classes=10,
                sample_size=(3, 32, 32),
            )
        ),
        "resnet101": ResNet(
            ResNetConfig(
                version=101,
                num_classes=10,
                sample_size=(3, 32, 32),
            )
        ),
    }

    for name, model in models.items():
        main(model, name)
