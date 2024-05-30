from dataclasses import dataclass
from pathlib import Path
from hashlib import md5
from zipfile import ZipFile
from urllib.request import urlretrieve
import shutil
import random
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from ._default import DefaultDataModuleConfig, DefaultDataModule


__all__ = ["TinyImageNetDataModuleConfig", "TinyImageNetDataModule"]


@dataclass
class TinyImageNetDataModuleConfig(DefaultDataModuleConfig):
    pass


class TinyImageNetDataModule(DefaultDataModule):
    def __init__(self, config: TinyImageNetDataModuleConfig):
        super().__init__(config)
        self.config = config

        self.dataset = datasets.ImageFolder

        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None

    def __repr__(self) -> str:
        return "tiny-imagenet"

    def get_num_classes(self):
        """Get the number of classes in the dataset.

        Returns:
            int: The number of classes in the dataset.
        """
        return 200

    def prepare_data(self):
        if not Path(self.config.data_dir + "/tiny-imagenet-200").exists():
            DATASET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
            DATASET_ZIP = Path(f".{self.config.data_dir}/tiny-imagenet-200.zip")
            DATASET_MD5_HASH = "90528d7ca1a48142e341f4ef8d21d0de"

            # Download Dataset if needed
            if not DATASET_ZIP.exists():
                print("Downloading the dataset, this may take a while...")

                with tqdm(
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    miniters=1,
                    desc=DATASET_URL.split("/")[-1],
                ) as t:

                    def show_progress(block_num, block_size, total_size):
                        t.total = total_size
                        t.update(block_num * block_size - t.n)

                    urlretrieve(
                        url=DATASET_URL, filename=DATASET_ZIP, reporthook=show_progress
                    )

            # Check MD5 Hash
            with DATASET_ZIP.open("rb") as f:
                assert (
                    md5(f.read()).hexdigest() == DATASET_MD5_HASH
                ), "The dataset zip file seems corrupted. Try to download it again."

            # Remove existing data set
            ORIGINAL_DATASET_DIR = Path(f".{self.config.data_dir}/original")
            if ORIGINAL_DATASET_DIR.exists():
                shutil.rmtree(ORIGINAL_DATASET_DIR)

            if not ORIGINAL_DATASET_DIR.exists():
                print("Extracting the dataset, this may take a while...")

                # Unzip the dataset
                with ZipFile(DATASET_ZIP, "r") as zip_ref:
                    for member in tqdm(zip_ref.infolist(), desc="Extracting"):
                        zip_ref.extract(member, ORIGINAL_DATASET_DIR)

            # Remove existing data set
            DATASET_DIR = Path(f".{self.config.data_dir}/tiny-imagenet-200")
            if DATASET_DIR.exists():
                shutil.rmtree(DATASET_DIR, ignore_errors=True)

            # Create the dataset directory
            if not DATASET_DIR.exists():
                print("Creating the dataset directory...")
                DATASET_DIR.mkdir()

            # Move train images to dataset directory
            ORIGINAL_TRAIN_DIR = ORIGINAL_DATASET_DIR / "tiny-imagenet-200" / "train"
            if ORIGINAL_TRAIN_DIR.exists():
                print("Moving train images...")
                ORIGINAL_TRAIN_DIR.replace(DATASET_DIR / "train")

            # Get validation images and annotations
            val_dict = {}
            ORIGINAL_VAL_DIR = ORIGINAL_DATASET_DIR / "tiny-imagenet-200" / "val"
            with (ORIGINAL_VAL_DIR / "val_annotations.txt").open("r") as f:
                for line in f.readlines():
                    split_line = line.split("\t")
                    if split_line[1] not in val_dict.keys():
                        val_dict[split_line[1]] = [split_line[0]]
                    else:
                        val_dict[split_line[1]].append(split_line[0])

            def split_list_randomly(
                input_list: list[str], split_ratio=0.5
            ) -> dict[str, list[str]]:
                # Shuffle the input list in-place
                random.shuffle(input_list)

                # Calculate the index to split the list
                split_index = int(len(input_list) * split_ratio)

                # Split the list into two parts
                return {
                    "val": input_list[:split_index],
                    "test": input_list[split_index:],
                }

            # Sample from validation images randomly into validation and test sets (50/50)
            print("Splitting original dataset images...")
            with tqdm(val_dict.items(), desc="Splitting images", unit="class") as t:
                for image_label, images in t:
                    for split_type, split_images in split_list_randomly(
                        images, split_ratio=0.5
                    ).items():
                        for image in split_images:
                            src = ORIGINAL_VAL_DIR / "images" / image
                            dest_folder = (
                                DATASET_DIR / split_type / image_label / "images"
                            )
                            dest_folder.mkdir(parents=True, exist_ok=True)
                            src.replace(dest_folder / image)
                    t.update()

            # Remove original directory
            shutil.rmtree(ORIGINAL_DATASET_DIR)

            # Remove original zip file
            DATASET_ZIP.unlink()

            print("Dataset is ready!")

        else:
            print("Dataset is already available!")

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = self.dataset(
                self.config.data_dir + "/tiny-imagenet-200/train",
                transform=self.config.transforms,
            )
            self.val_dataset = self.dataset(
                self.config.data_dir + "/tiny-imagenet-200/val",
                transform=self.config.transforms,
            )
            self.test_dataset = self.dataset(
                self.config.data_dir + "/tiny-imagenet-200/test",
                transform=self.config.transforms,
            )

    def train_dataloader(self):
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
        )

    def val_dataloader(self):
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
        )

    def test_dataloader(self):
        assert self.test_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
        )


if __name__ == "__main__":
    dmc = TinyImageNetDataModuleConfig(
        data_dir="data",
        batch_size=32,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        transforms=transforms.Compose([transforms.ToTensor()]),
    )
    datamodule = TinyImageNetDataModule(dmc)
    datamodule.prepare_data()
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
