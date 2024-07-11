from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\\n" + fh.read()

setup(
    name="ml_zoo",
    version=f'{{ VERSION_PLACEHOLDER }}'
    author="Karan Ravindra",
    author_email="contact@karanravindra.com",
    description="A collection of machine learning utilities and models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/karanravindra/ml-zoo",
    packages=find_packages(exclude=["tests"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "setuptools",
        "wheel",
        "torch",
        "torchvision",
        "lightning",
        "einops",
        "wandb",
    ],
    extras_require={
        "dev": [
            "pytest",
            "ruff",
            "pre-commit",
        ],
    },
    python_requires=">=3.8",
)
