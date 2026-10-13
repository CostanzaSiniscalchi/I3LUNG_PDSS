import os
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')


import torch
import pandas as pd
import numpy as np
import numpy.typing as npt
from typing import List, Optional, Union, Tuple
from sklearn.preprocessing import OneHotEncoder
from sklearn import __version__ as sklearn_version
from packaging import version
from fastai.vision.all import (
    DataLoaders, Learner, SaveModelCallback, CSVLogger, Callback
)
from .._params import BaseMultimodalLoss, TrainerConfig

from MIL.util import log
from MIL.model import torch_utils

# -----------------------------------------------------------------------------

class MultimodalLossLogger(Callback):
    """Write per-epoch classification and reconstruction sub-losses to subloss_history.csv.

    Reconstruction loss is saved unweighted (before multiplication by reconstruction_weight)
    so the raw component losses are interpretable independently of the hyperparameter.
    """

    order = CSVLogger.order + 1  # run after CSVLogger so epoch number is already written

    def before_fit(self):
        self._is_multimodal = isinstance(self.learn.loss_func, BaseMultimodalLoss)
        if self._is_multimodal:
            self._reset_buffers()
            outdir = str(self.learn.path)
            self._csv_path = os.path.join(outdir, 'subloss_history.csv')
            if not os.path.exists(self._csv_path):
                with open(self._csv_path, 'w') as f:
                    f.write('epoch,train_classification_loss,train_reconstruction_loss,'
                            'valid_classification_loss,valid_reconstruction_loss\n')

    def _reset_buffers(self):
        self._train_cls, self._train_rec = [], []
        self._val_cls, self._val_rec = [], []

    def after_batch(self):
        if not self._is_multimodal:
            return
        lf = self.learn.loss_func
        cls = getattr(lf, 'last_classification_loss', None)
        rec = getattr(lf, 'last_reconstruction_loss', None)
        if cls is None or rec is None:
            return
        if self.training:
            self._train_cls.append(cls)
            self._train_rec.append(rec)
        else:
            self._val_cls.append(cls)
            self._val_rec.append(rec)

    def after_epoch(self):
        if not self._is_multimodal or not self._train_cls:
            return
        row = (
            f"{self.epoch},"
            f"{np.mean(self._train_cls):.6f},"
            f"{np.mean(self._train_rec):.6f},"
            f"{np.mean(self._val_cls) if self._val_cls else float('nan'):.6f},"
            f"{np.mean(self._val_rec) if self._val_rec else float('nan'):.6f}\n"
        )
        with open(self._csv_path, 'a') as f:
            f.write(row)
        self._reset_buffers()

# -----------------------------------------------------------------------------

class SaveFirstEpochCallback(Callback):
    """Save a checkpoint after the first training epoch, independent of
    whether it is the best-performing epoch so far (which ``SaveModelCallback``
    would otherwise overwrite as later epochs improve)."""

    order = 60  # run after SaveModelCallback

    def after_epoch(self):
        if self.epoch == 0:
            self.learn.save("epoch_1")

# -----------------------------------------------------------------------------

def train(learner, config, callbacks=None):
    """Train an attention-based multi-instance learning model with FastAI.

    Args:
        learner (``fastai.learner.Learner``): FastAI learner.
        config (``TrainerConfig``): Trainer and model configuration.

    Keyword args:
        callbacks (list(fastai.Callback)): FastAI callbacks. Defaults to None.
    """
    cbs = [
        SaveModelCallback(fname=f"best_valid", monitor=config.save_monitor),
        SaveFirstEpochCallback(),
        CSVLogger(),
        MultimodalLossLogger(),
    ]
    if callbacks:
        cbs += callbacks
    if config.fit_one_cycle:
        if config.lr is None:
            lr = learner.lr_find().valley
            log.info(f"Using auto-detected learning rate: {lr}")
        else:
            lr = config.lr
        learner.fit_one_cycle(n_epoch=config.epochs, lr_max=lr, cbs=cbs)
    else:
        if config.lr is None:
            lr = learner.lr_find().valley
            log.info(f"Using auto-detected learning rate: {lr}")
        else:
            lr = config.lr
        learner.fit(n_epoch=config.epochs, lr=lr, wd=config.wd, cbs=cbs)
    return learner

# -----------------------------------------------------------------------------

def build_learner(
    config: TrainerConfig,
    bags: List[str],
    targets: npt.NDArray,
    train_idx: npt.NDArray[np.int_],
    val_idx: npt.NDArray[np.int_],
    unique_categories: npt.NDArray,
    outdir: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    **dl_kwargs
) -> Tuple[Learner, Tuple[int, int]]:
    """Build a FastAI learner for training an MIL model.

    Args:
        config (``TrainerConfig``): Trainer and model configuration.
        bags (list(str)): Path to .pt files (bags) with features, one per patient.
        targets (np.ndarray): Category labels for each patient, in the same
            order as ``bags``.
        train_idx (np.ndarray, int): Indices of bags/targets that constitutes
            the training set.
        val_idx (np.ndarray, int): Indices of bags/targets that constitutes
            the validation set.
        unique_categories (np.ndarray(str)): Array of all unique categories
            in the targets. Used for one-hot encoding.
        outdir (str): Location in which to save training history and best model.
        device (torch.device or str): PyTorch device.

    Returns:
        fastai.learner.Learner, (int, int): FastAI learner and a tuple of the
            number of input features and output classes.

    """
    log.debug("Building FastAI learner")

    # Prepare device.
    device = torch_utils.get_device(device)

    # Prepare data.
    # Set oh_kw to a dictionary of keyword arguments for OneHotEncoder,
    # using the argument sparse=False if the sklearn version is <1.2
    # and sparse_output=False if the sklearn version is >=1.2.
    if version.parse(sklearn_version) < version.parse("1.2"):
        oh_kw = {"sparse": False}
    else:
        oh_kw = {"sparse_output": False}

    if config.model_type in ['classification', 'multimodal']:
        encoder = OneHotEncoder(**oh_kw).fit(unique_categories.reshape(-1, 1))
    else:
        encoder = None

    # Build the dataloaders.
    train_dl = config.build_train_dataloader(
        bags[train_idx],
        targets[train_idx],
        encoder=encoder,
        dataloader_kwargs=dict(
            num_workers=1,
            device=device,
            pin_memory=True,
            **dl_kwargs
        )
    )
    val_dl = config.build_val_dataloader(
        bags[val_idx],
        targets[val_idx],
        encoder=encoder,
        dataloader_kwargs=dict(
            shufle=False,
            num_workers=8,
            persistent_workers=True,
            device=device,
            pin_memory=False,
            **dl_kwargs
        )
    )

    # Prepare model.
    batch = train_dl.one_batch()
    n_in, n_out = config.inspect_batch(batch)
    n_out = 1 if config.model_type in ['survival', 'multimodal_survival'] else n_out
    model = config.build_model(n_in, n_out).to(device)

    if hasattr(model, 'relocate'):
        model.relocate()

    # Loss should weigh inversely to class occurences.


    if config.model_type in ['classification', 'multimodal'] and config.weighted_loss:
        #counts = pd.value_counts(targets[train_idx])
        counts = pd.Series(targets[train_idx]).value_counts()
        weights = counts.sum() / counts
        weights /= weights.sum()
        weights = torch.tensor(
            list(map(weights.get, encoder.categories_[0])), dtype=torch.float32
        ).to(device)
        loss_kw = {"weight": weights}
    else:
        loss_kw = {}
    if config.model_type in ['multimodal', 'multimodal_survival'] and config.reconstruction_weight:
        loss_kw['reconstruction_weight'] = config.reconstruction_weight
    loss_func = config.loss_fn(**loss_kw)

    # Create learning and fit.
    dls = DataLoaders(train_dl, val_dl)
    learner = Learner(dls, model, loss_func=loss_func, metrics=config.get_metrics(), path=outdir)

    return learner, (n_in, n_out)
