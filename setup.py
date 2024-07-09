import codecs
import os.path
from setuptools import setup, find_packages


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="ml_zoo",
    version=get_version("ml_zoo/__init__.py"),
    author="Karan Ravindra",
    author_email="contact@karanravindra.com",
    description="A collection of machine learning utilities and models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/karanravindra/ml-zoo",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    requires=[
        "torch",
        "torchvision",
        # "pytorch-lightning",
        # "wandb",
    ],
    install_requires=[
        "setuptools",
        "wheel",
    ],
    extras_require={
        "dev": [
            "pytest",
            "ruff"
        ],
    },
    python_requires=">=3.8",
)
