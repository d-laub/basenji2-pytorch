# Basenji2 in PyTorch

This repo provides a PyTorch re-implementation of the Basenji2 model published in ["Cross-species regulatory sequence activity prediction"](https://doi.org/10.1371/journal.pcbi.1008050) by David Kelley.

## Installation

On Linux with conda/mamba:

1. Clone the repository.
2. Add it to your PYTHONPATH environment variable (i.e. in your `.bashrc` file).
3. Use conda/mamba to install dependencies from the `environment.yml` found in the repo.
4. Download the [PyTorch weights](https://drive.google.com/file/d/1VrSyIOqcdIMbgstsJztn5P8a9h7p0Dnb/view?usp=sharing).

## Usage

```python
import json
import torch
from basenji_pytorch import Basenji2, params # or PLBasenji2 to use training parameters from Kelley et al. 2020

model_weights = 'path/to/basenji2.pth'

with open(params) as params_open:
    model_params = json.load(params_open)['model']

# to use a headless model e.g. for transfer learning
# model_params.pop("head_human", None)

basenji2 = Basenji2(model_params)
basenji2.load_state_dict(torch.load(model_weights), strict=False)
```
