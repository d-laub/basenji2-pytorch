# Basenji2 in PyTorch

This repo provides a PyTorch re-implementation of the Basenji2 model published in ["Cross-species regulatory sequence activity prediction"](https://doi.org/10.1371/journal.pcbi.1008050) by David Kelley. This implementation was checked by verifying that the Tensorflow and PyTorch version yielded the same output on random data. Small deviations were found, likely due to differences in the underlying algorithms used by Tensorflow and PyTorch (e.g. different matrix multiplication algorithms). In addition, [Qixiu Du kindly computed evaluation metrics](https://github.com/d-laub/basenji2-pytorch/issues/1) and found that the PyTorch re-implementation achieves competitive performance on real data, further validating the port.

## Installation

pip install basenji2-pytorch

## Usage

```python
import torch
from basenji2_pytorch import Basenji2, basenji2_params, basenji2_weights # or PLBasenji2 to also use training parameters from Kelley et al. 2020

# to use a headless model e.g. for transfer learning
# basenji2_params["model"].pop("head_human", None)

basenji2 = Basenji2(basenji2_params["model"])
basenji2.load_state_dict(torch.load(basenji2_weights()), strict=False)
```

- `basenji2_params` is a dictionary of both training and model parameters matching the implementation in Kelley et al. 2020
- `basenji2_weights` is a function that uses [pooch](https://github.com/fatiando/pooch) to download weights from Zenodo and return the path as a string.
- `Basenji2` is a PyTorch nn.Module that can be initialized from the model parameters of `basenji2_params`
- `PLBasenji2` is a PyTorch Lightning module that can be initialized from `basenji2_params` to match both the training and architectural parameters of Basenji2