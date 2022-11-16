# Usage

```python
import json
import torch
from basenji_pytorch import Basenji2 # or PLBasenji2 to use training parameters from Kelley et al. 2020

params_file = 'path/to/params_human.json'
model_weights = 'path/to/basenji2.pth'

with open(params_file) as params_open:
    model_params = json.load(params_open)['model']

# to use a headless model e.g. for transfer learning
# model_params.pop("head_human", None)

basenji2 = Basenji2(model_params)
basenji2.load_state_dict(torch.load(model_weights), strict=False)
```
