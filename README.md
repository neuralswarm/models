# Neural Swarm Model Zoo

The Neural Swarm Model Zoo is a collection of Machine Learning models written in [PyTorch](https://pytorch.org) and [PyTorch Lightning](https://www.pytorchlightning.ai).

Accompanying each model are [Model Cards](https://arxiv.org/pdf/1810.03993.pdf) in the `cards` folder. Model Cards provide additional details such as:

- Carbon emissions
- Intended use
- Loss
- Metric
- Parameters

Datasets are from [Hugging Face Hub](https://huggingface.co/datasets).

## Models

| Task | Model | Year | Dataset | Paper |
|-|-|-|-|-|
| Image Classification | LeNet | 1998 | [Dataset](https://huggingface.co/datasets/mnist) | [Paper](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf) |

## Usage

**Clone repository**
```bash
git clone https://github.com/neuralswarm/models.git
cd models
```

**Install dependencies**
```bash
poetry install
```

**Train/test model**
```bash
poetry run python models/le_net.py
```

## License
The Neural Swarm Model Zoo is released under the MIT license, as found in [LICENSE](https://github.com/neuralswarm/models/blob/master/LICENSE).