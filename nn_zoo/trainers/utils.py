import torch


def get_optim(
    optim: str,
) -> type[torch.optim.optimizer.Optimizer]:  # TODO: Add all the optimizer types
    """A function to get the optimizer class from the string
    Supports all the optimizers availible in PyTorch

    Args:
        optim (str): The optimizer name

    Raises:
        NotImplementedError: If the requested optimizer is not availible

    Returns:
        type[torch.optim.optimizer.Optimizer]: The optimizer class
    """
    match optim.lower():
        case "sgd":
            return torch.optim.SGD
        case "adam":
            return torch.optim.Adam
        case "adamw":
            return torch.optim.AdamW
        case _:
            raise NotImplementedError(
                f"The requested optimizer: {optim} is not availible"
            )


def get_scheduler(
    scheduler: str,
) -> type[torch.optim.lr_scheduler.LRScheduler]:  # TODO: Add all the scheduler types
    """A function to get the scheduler class from the string
    Supports all the schedulers availible in PyTorch

    Args:
        scheduler (str): The scheduler name

    Raises:
        NotImplementedError: If the requested scheduler is not availible

    Returns:
        type[torch.optim.lr_scheduler.LRScheduler]: The scheduler class
    """
    match scheduler.lower():
        case "steplr":
            return torch.optim.lr_scheduler.StepLR
        case "multisteplr":
            return torch.optim.lr_scheduler.MultiStepLR
        case "exponentiallr":
            return torch.optim.lr_scheduler.ExponentialLR
        case "cosinelr":
            return torch.optim.lr_scheduler.CosineAnnealingLR
        case _:
            raise NotImplementedError(
                f"The requested scheduler: {scheduler} is not availible"
            )
