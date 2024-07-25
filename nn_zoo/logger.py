from typing import Optional, Literal
import torch
from torch import nn

import wandb
from model import Model


class WandbLogger:
    def __init__(self, project_name, run_name, config):
        wandb.init(project=project_name, name=run_name, config=config)
        self.config = config

    def log(self, name, metrics):
        wandb.log({name: metrics}, commit=True)

    def finish(self):
        wandb.finish()

    def watch(
        self,
        model: nn.Module,
        criterion=None,
        log: Optional[Literal["gradients", "parameters", "all"]] = "gradients",
        log_freq: int = 1000,
        idx: Optional[int] = None,
        log_graph: bool = (False),
    ):
        wandb.watch(
            model,
            criterion=criterion,
            log=log,
            log_freq=log_freq,
            idx=idx,
            log_graph=log_graph,
        )


if __name__ == "__main__":
    logger = WandbLogger("nn_zoo", "default", {"model": Model, "dm": "DataModule"})
    model = torch.nn.Linear(1, 1)
    logger.watch(model)

    optim = torch.optim.Adam(model.parameters())

    for i in range(1000):
        optim.zero_grad()
        loss = model(torch.randn(1, 1))
        loss.backward()
        optim.step()
        if i % 100 == 0:
            logger.log("loss", loss.item())
    logger.finish()
