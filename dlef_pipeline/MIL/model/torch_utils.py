from typing import Optional

import torch


def get_device(device: Optional[str] = None):
    if device is None and torch.cuda.is_available():
        return torch.device('cuda')
    elif (device is None
          and hasattr(torch.backends, 'mps')
          and torch.backends.mps.is_available()):
        return torch.device('mps')
    elif device is None:
        return torch.device('cpu')
    elif isinstance(device, str):
        return torch.device(device)
    else:
        return device