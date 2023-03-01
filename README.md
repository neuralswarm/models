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

**Train model**
```bash
poetry run python models/le_net.py fit \
    --trainer.accelerator 'gpu' \
    --trainer.devices 2 \
    --trainer.strategy 'ddp' \
    --trainer.max_epochs 10 \
    --trainer.callbacks+=ModelSummary \
    --trainer.callbacks.max_depth=2 \
    --data.batch_size 128 \
    --data.num_workers 4 \
    --data.pin_memory true
```

**Evaluate model**
```bash
poetry run python models/le_net.py test \
    --data.batch_size 128 \
    --data.num_workers 4 \
    --data.pin_memory true \
    --ckpt_path weights/le_net.ckpt
```

## License
The Neural Swarm Model Zoo is released under the MIT license, as found in [LICENSE](https://github.com/neuralswarm/models/blob/master/LICENSE).