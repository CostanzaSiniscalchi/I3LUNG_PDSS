"""Dataset utility functions for MIL."""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Callable, Union, Protocol
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset
from functools import partial

# -----------------------------------------------------------------------------

def _zip(bag_data, targets, encoder=None):
    _targets = targets if encoder is None else targets.squeeze()
    return (*bag_data, _targets)

def build_multimodal_mixed_bag_dataset(
        bags, targets, encoder, bag_size,
        use_lens=False, max_bag_size=None,
        dtype=torch.float32):

    assert len(bags) == len(targets)

    zip_fn = partial(_zip, encoder=encoder)

    dataset = MapDataset(
        zip_fn,
        MixedMultiBagDataset(bags, dtype=dtype),
        EncodedDataset(encoder, targets),
    )
    dataset.encoder = encoder
    return dataset


# -----------------------------------------------------------------------------

@dataclass
class MultiBagDataset(Dataset):
    """A dataset of bags of instances, with multiple bags per instance."""

    bags: List[Union[List[Path], List[np.ndarray], List[torch.Tensor], List[List[str]]]]
    """Bags for each slide.

    This can either be a list of `.pt` files, a list of numpy arrays, a list
    of Tensors, or a list of lists of strings (where each item in the list is
    a patient, and nested items are slides for that patient).

    Each bag consists of features taken from all images from a slide. Data
    should be of shape N x F, where N is the number of instances and F is the
    number of features per instance/slide.
    """

    n_bags: int
    """Number of bags per instance."""

    bag_size: Optional[int] = None
    """The number of instances in each bag.
    For bags containing more instances, a random sample of `bag_size`
    instances will be drawn.  Smaller bags are padded with zeros.  If
    `bag_size` is None, all the samples will be used.
    """

    max_bag_size: Optional[int] = None

    dtype: torch.dtype = torch.float32

    def __post_init__(self):
        if self.bag_size and self.max_bag_size:
            raise ValueError("Cannot specify both bag_size and max_bag_size")

    def __len__(self):
        return len(self.bags)


# -----------------------------------------------------------------------------

class MapDataset(Dataset):
    def __init__(
        self,
        func: Callable,
        *datasets: Union[npt.NDArray, Dataset],
        strict: bool = True
    ) -> None:
        """A dataset mapping over a function over other datasets.
        Args:
            func:  Function to apply to the underlying datasets.  Has to accept
                `len(dataset)` arguments.
            datasets:  The datasets to map over.
            strict:  Enforce the datasets to have the same length.  If
                false, then all datasets will be truncated to the shortest
                dataset's length.
        """
        if strict:
            assert all(len(ds) == len(datasets[0]) for ds in datasets)  # type: ignore
            self._len = len(datasets[0])  # type: ignore
        elif datasets:
            self._len = min(len(ds) for ds in datasets)  # type: ignore
        else:
            self._len = 0

        self._datasets = datasets
        self.func = func
        self.encoder = None

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index: int) -> Any:
        return self.func(*[ds[index] for ds in self._datasets])

    def new_empty(self):
        # FIXME hack to appease fastai's export
        return self

# -----------------------------------------------------------------------------

class SKLearnEncoder(Protocol):
    """An sklearn-style encoder."""

    categories_: List[List[str]]

    def transform(self, x: List[List[Any]]):
        ...


# -----------------------------------------------------------------------------

class EncodedDataset(MapDataset):
    def __init__(self, encode: SKLearnEncoder, values: npt.NDArray):
        """A dataset which first encodes its input data.
        This class is can be useful with classes such as fastai, where the
        encoder is saved as part of the model.
        Args:
            encode:  an sklearn encoding to encode the data with.
            values:  data to encode.
        """
        super().__init__(self._unsqueeze_to_float32, values)
        self.encode = encode

    def _unsqueeze_to_float32(self, x):
        if self.encode is None:
            return torch.tensor(x, dtype=torch.float32)
        return torch.tensor(
            self.encode.transform(np.array(x).reshape(1, -1)), dtype=torch.float32
        )

# -----------------------------------------------------------------------------

class MixedMultiBagDataset(Dataset):
    """Dataset for mixed multimodal bags where some modalities may be missing."""

    def __init__(
        self,
        bags: List[str],
        bag_size: Optional[int] = None,
        max_bag_size: Optional[int] = None,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        if bag_size and max_bag_size:
            raise ValueError("Cannot specify both bag_size and max_bag_size")
        self.bags = bags
        self.bag_size = bag_size
        self.max_bag_size = max_bag_size
        self.dtype = dtype

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, index: int):
        # Load the multimodal bag dictionary
        bag_dict = torch.load(self.bags[index], weights_only=True) #wight_only=True for torch 2.0+
        
        # Get the modality mask
        mask = bag_dict['mask']
        
        # Process each feature modality
        processed_features = []
        for i in range(1, len(mask) + 1):  # Use mask length to determine number of features
            feat_key = f'feature{i}'
            features = bag_dict[feat_key].to(self.dtype)

            processed_features.append(features)

        return (*processed_features, mask)