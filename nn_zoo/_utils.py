import torch
from model import Model


def get_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)

    device = "cpu"
    if torch.backends.cudnn.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"

    return torch.device(device)


def get_optimizer(optim: str) -> type[torch.optim.Optimizer]:
    match optim:
        case "adadelta":
            return torch.optim.Adadelta
        case "adagrad":
            return torch.optim.Adagrad
        case "adam":
            return torch.optim.Adam
        case "adamax":
            return torch.optim.Adamax
        case "adamw":
            return torch.optim.AdamW
        case "asgd":
            return torch.optim.ASGD
        case "lbfgs":
            return torch.optim.LBFGS
        case "nadam":
            return torch.optim.NAdam
        case "radam":
            return torch.optim.RAdam
        case "rmsprop":
            return torch.optim.RMSprop
        case "rprop":
            return torch.optim.Rprop
        case "sgd":
            return torch.optim.SGD
        case "sparse_adam":
            return torch.optim.SparseAdam
        case _:
            raise ValueError(f"Unknown optimizer: {optim}")


def get_scheduler(sch: str) -> type[torch.optim.lr_scheduler.LRScheduler]:
    match sch:
        case "cosine_annealing_lr":
            return torch.optim.lr_scheduler.CosineAnnealingLR
        case "cosine_annealing_warm_restarts":
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
        case "cyclic_lr":
            return torch.optim.lr_scheduler.CyclicLR
        case "exponential_lr":
            return torch.optim.lr_scheduler.ExponentialLR
        case "lambda_lr":
            return torch.optim.lr_scheduler.LambdaLR
        case "multiplicative_lr":
            return torch.optim.lr_scheduler.MultiplicativeLR
        case "one_cycle_lr":
            return torch.optim.lr_scheduler.OneCycleLR
        case "reduce_lr_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau
        case "step_lr":
            return torch.optim.lr_scheduler.StepLR
        case _:
            raise ValueError(f"Unknown scheduler: {sch}")


def summary(
    model: Model,
    input_size: list[int] | tuple,
    batch_dim: bool = True,
    dtype: torch.dtype = torch.float32,
) -> None:
    input_tensor = torch.rand(*input_size, dtype=dtype)
    if batch_dim:
        input_tensor = input_tensor.unsqueeze(0)

    # # depth = 1 # TODO Implement depth
    def hook_fn(module, input, output):
        print(module.__class__.__name__, tuple(output.shape))

    hooks = []
    for layer in model.children():
        hook = layer.register_forward_hook(hook_fn)
        hooks.append(hook)

    model(input_tensor)

    for hook in hooks:
        hook.remove()
