import tomllib

with open("pyproject.toml", "rb") as f:
    pyproject = tomllib.load(f)

__version__ = pyproject["project"]["version"]
