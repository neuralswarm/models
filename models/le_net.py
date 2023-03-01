import collections

import pytorch_lightning as pl
import torch
import torch.nn as nn
from codecarbon import OfflineEmissionsTracker
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.cli import LightningCLI
from torch import optim
from torch.nn.common_types import _size_2_t
from torchmetrics import Accuracy

from data.mnist import MNISTDataModule


class Conv2dMap(nn.Module):
    """Applies a 2D convolution with a specific mapping of input to output channels.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        mapping (Tensor): A 2D tensor of shape (in_channels, out_channels) where each
            row is a binary vector indicating which input channels are used to compute
            the corresponding output channel.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        mapping: torch.LongTensor,
    ):
        super().__init__()
        assert (in_channels, out_channels) == mapping.shape
        self.convs = nn.ModuleList(
            [nn.Conv2d(i.item(), 1, kernel_size) for i in mapping.sum(dim=0)]
        )
        self.register_buffer("mapping", mapping)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor):
        x_is = []
        for i in range(self.out_channels):
            channel_indexes = self.mapping[:, i].nonzero().squeeze()  # type: ignore
            x_i = x.index_select(dim=1, index=channel_indexes)
            x_i = self.convs[i](x_i)
            x_is.append(x_i)
        return torch.cat(x_is, dim=1)


class RadialBasisFunction(nn.Module):
    """Computes the sum squared distances between a tensor and a set of basis vectors.

    Args:
        in_features (int): Size of each input sample
        out_features (int): Size of each output sample
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, in_features, out_features))

    def forward(self, x: torch.Tensor):
        return (x.unsqueeze(-1) - self.weight).pow(2).sum(dim=1)


class SumPool2D(nn.Module):
    """Applies a scaled 2D sum pooling over an input signal composed of input planes.

    Args:
        channels (int): Number of channels in the input
        kernel_size: Size of the window
    """

    def __init__(self, channels: int, kernel_size: _size_2_t):
        super().__init__()
        self.bias = nn.Parameter(torch.rand(1, channels, 1, 1))
        self.pool = nn.AvgPool2d(kernel_size, divisor_override=1)
        self.weight = nn.Parameter(torch.rand(1, channels, 1, 1))

    def forward(self, x: torch.Tensor):
        return self.weight * self.pool(x) + self.bias


class TanhScale(nn.Module):
    """Applies a scaled Hyperbolic Tangent (Tanh) function element-wise.

    Args:
        A (float): Amplitude of the function
        S (float): Slope at the origin
    """

    def __init__(self, A: float = 1.7159, S: float = 2 / 3):
        super().__init__()
        self.A = A
        self.S = S

    def forward(self, x: torch.Tensor):
        return self.A * torch.tanh(self.S * x)


class LeNet(pl.LightningModule):
    """LeNet-5 convolutional neural network model used for MNIST digit classification.

    Based on `Gradient-based Learning Applied to Document Recognition` - Lecun, Y.,
    Bottou, L., Bengio, Y. & Haffner, P. (1998)
    """

    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.Tensor(32, 1, 28, 28)
        self.accuracy = Accuracy(task="multiclass", num_classes=10)
        self.mapping = torch.LongTensor(
            [
                [1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1],
                [1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1],
                [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1],
                [0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1],
                [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1],
                [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1],
            ]
        )
        self.model = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "C1",
                        nn.Conv2d(
                            in_channels=1,
                            out_channels=6,
                            kernel_size=5,
                            padding="same",
                        ),
                    ),
                    ("A1", TanhScale()),
                    ("S2", SumPool2D(channels=6, kernel_size=2)),
                    ("A2", TanhScale()),
                    (
                        "C3",
                        Conv2dMap(
                            in_channels=6,
                            out_channels=16,
                            kernel_size=5,
                            mapping=self.mapping,
                        ),
                    ),
                    ("A3", TanhScale()),
                    ("S4", SumPool2D(channels=16, kernel_size=2)),
                    ("A4", TanhScale()),
                    ("C5", nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)),
                    ("A5", TanhScale()),
                    ("FLATTEN", nn.Flatten()),
                    ("F6", nn.Linear(in_features=120, out_features=84)),
                    ("A6", TanhScale()),
                    ("OUTPUT", RadialBasisFunction(in_features=84, out_features=10)),
                ]
            )
        )

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def on_fit_start(self):
        self.tracker = OfflineEmissionsTracker(country_iso_code="USA")
        self.tracker.start()

    def on_fit_end(self):
        self.tracker.stop()
        self.trainer.save_checkpoint("weights/le_net.ckpt", weights_only=True)

    def training_step(self, batch, batch_idx):
        metrics = self._shared_eval_step(batch, batch_idx)
        return metrics["train_loss"]

    def validation_step(self, batch, batch_idx):
        metrics = self._shared_eval_step(batch, batch_idx)
        return metrics

    def test_step(self, batch, batch_idx):
        metrics = self._shared_eval_step(batch, batch_idx)
        return metrics

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        x = x.unsqueeze(1).float()
        y_hat = self(x)
        stage = self.trainer.state.stage
        metrics = {
            f"{stage}_loss": torch.gather(y_hat, 1, y.unsqueeze(-1)).mean(),
            f"{stage}_metric": self.accuracy(-y_hat, y),
        }
        self.log_dict(metrics)
        return metrics

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch["image"], batch["label"]
        x = x.unsqueeze(1).float()
        y_hat = self(x)
        return y_hat

    def configure_callbacks(self):
        early_stopping = EarlyStopping(monitor="validate_loss", mode="min")
        return [early_stopping]

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def cli_main():
    cli = LightningCLI(LeNet, MNISTDataModule)  # noqa: F841


if __name__ == "__main__":
    cli_main()
