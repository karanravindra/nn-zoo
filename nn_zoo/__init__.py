import tomllib

from . import datamodules, models, trainers

with open("pyproject.toml", "rb") as f:
    pyproject = tomllib.load(f)

__name__ = "nn_zoo"
__version__ = pyproject["project"]["version"]
__all__ = ["datamodules", "models", "trainers"]
