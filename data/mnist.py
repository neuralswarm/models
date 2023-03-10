import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils import data


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str):
        if stage == "fit" or "validate":
            train_val_dataset = load_dataset("mnist", split="train")
            train_val_dataset.set_format(type="torch")
            self.train_dataset, self.val_dataset = data.random_split(
                train_val_dataset, [0.8, 0.2]
            )

        if stage == "test":
            self.test_dataset = load_dataset("mnist", split="test")
            self.test_dataset.set_format(type="torch")

        if stage == "predict":
            self.predict_dataset = load_dataset("mnist", split="test")
            self.predict_dataset.set_format(type="torch")

    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, **self.hparams)

    def val_dataloader(self):
        return data.DataLoader(self.val_dataset, **self.hparams)

    def test_dataloader(self):
        return data.DataLoader(self.test_dataset, **self.hparams)

    def predict_dataloader(self):
        return data.DataLoader(self.predict_dataset, **self.hparams)
