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
    CIFARDataModule,
    CIFARDataModuleConfig,
    Classifier,
    ClassifierConfig,
    VGG,
    VGGConfig,
)


def main(model: nn.Module, run_name: str = "qmnist"):
    # Create DataModule
    dm_config = CIFARDataModuleConfig(
        data_dir="data",
        batch_size=128,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        transforms=[
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2467, 0.2432, 0.2612)),
        ],
        use_cifar100=False,
    )

    dm = CIFARDataModule(dm_config)

    # Create Classifier
    classifier_config = ClassifierConfig(
        model=model,
        dm=dm,
        optim="SGD",
        optim_args={"lr": 0.01, "momentum": 0.9},
        # scheduler="CosineAnnealingLR",
        # scheduler_args={"T_max": 100},
    )

    classifier = Classifier(classifier_config)

    # Log model
    logger = WandbLogger(
        name=run_name,
        project="cifar",
        dir="blog/3-cifar/logs",
        save_dir="blog/3-cifar/logs",
        log_model=True,
    )

    # Train
    trainer = pl.Trainer(
        logger=logger,
        default_root_dir="blog/3-cifar/logs",
        max_epochs=100,
        check_val_every_n_epoch=1,
        enable_model_summary=False,
    )

    logger.watch(model, log="all", log_freq=100, log_graph=True)

    summary(model, input_size=(1, 3, 32, 32))

    trainer.fit(classifier)
    trainer.test(classifier)

    logger.experiment.finish()


if __name__ == "__main__":
    models = {
        # "lenet 5": LeNet(
        #     LeNetConfig(
        #         sample_size=(3, 32, 32),
        #         version=5,
        #         activation="ReLU",
        #         poolings="MaxPool2d",
        #     )
        # )   ,
        # "lenet 5 drop2": LeNet(
        #     LeNetConfig(
        #         sample_size=(3, 32, 32),
        #         version=5,
        #         activation="ReLU",
        #         poolings="MaxPool2d",
        #         dropouts=[0.5, 0.5, 0],
        #     )
        # )   ,
        # "lenet 5 drop2 m": LeNet(
        #     LeNetConfig(
        #         sample_size=(3, 32, 32),
        #         version=None,
        #         feature_dims=[3, 12, 32],
        #         vectors=[32 * 5 * 5, 120, 84, 10],
        #         activation="ReLU",
        #         poolings="MaxPool2d",
        #         dropouts=[0.5, 0.5, 0],
        #     )
        # )   ,
        # "lenet 5 drop2 l": LeNet(
        #     LeNetConfig(
        #         sample_size=(3, 32, 32),
        #         version=None,
        #         feature_dims=[3, 24, 64],
        #         vectors=[64 * 5 * 5, 120, 84, 10],
        #         activation="ReLU",
        #         poolings="MaxPool2d",
        #         dropouts=[0.5, 0.5, 0],
        #     )
        # )   ,
        # "lenet 5 drop2 xl": LeNet(
        #     LeNetConfig(
        #         sample_size=(3, 32, 32),
        #         version=None,
        #         feature_dims=[3, 32, 128],
        #         vectors=[128 * 5 * 5, 120, 84, 10],
        #         activation="ReLU",
        #         poolings="MaxPool2d",
        #         dropouts=[0.5, 0.5, 0],
        #     )
        # )   ,
        # "cnn": nn.Sequential(
        #     nn.Conv2d(3, 32, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(32, 64, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Flatten(),
        #     nn.Dropout(0.5),
        #     nn.Linear(128 * 4 * 4, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 10),
        # ),
        # "cnn lg": nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(128, 256, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(256, 512, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Flatten(),
        #     nn.Dropout(0.5),
        #     nn.Linear(512 * 2 * 2, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 10),
        # ),
        "vgg 11": VGG(
            VGGConfig(
                version=11,
                sample_size=(3, 32, 32),
                vector_dim=512, # 4096 -> 512
                num_classes=10,
                global_pooling_dim=1
            )
        ),
        # "vgg 13": VGG(
        #     VGGConfig(
        #         version=13,
        #         sample_size=(3, 32, 32),
        #         vector_dim=4096, # 4096 -> 512
        #         num_classes=10,
        #         global_pooling_dim=1
        #     )
        # ),
        # "vgg 16": VGG(
        #     VGGConfig(
        #         version=16,
        #         sample_size=(3, 32, 32),
        #         vector_dim=4096, # 4096 -> 512
        #         num_classes=10,
        #         global_pooling_dim=1
        #     )
        # ),
        # "vgg 19": VGG(
        #     VGGConfig(
        #         version=19,
        #         sample_size=(3, 32, 32),
        #         vector_dim=4096, # 4096 -> 512
        #         num_classes=10,
        #         global_pooling_dim=1
        #     )
        # ),
    }

    for name, model in models.items():
        main(model, name)
