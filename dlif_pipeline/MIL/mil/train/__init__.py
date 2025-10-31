"""Training functions for various multi-instance learning (MIL) models."""

import numpy as np
from os.path import join, exists
from typing import Union, List, Optional, TYPE_CHECKING
from MIL.util import path_to_name, log, load_json, write_json
from MIL.dataset import Dataset
from os.path import join, isdir

from .. import utils
from ..eval import predict_mil
from .._params import TrainerConfig

if TYPE_CHECKING:
    from fastai.learner import Learner


# -----------------------------------------------------------------------------

def train_mil(
    config: TrainerConfig,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset],
    outcomes: Union[str, List[str]],
    bags: Union[str, List[str]],
    *,
    outdir: str = 'mil',
    exp_label: Optional[str] = None,
    **kwargs
) -> "Learner":
    """Train a multiple-instance learning (MIL) model.

    This high-level trainer facilitates training from a given MIL configuration,
    using Datasets as input and with input features taken from a given directory
    of bags.

    Args:
        config (:class:`MIL.mil.TrainerConfig`):
            Trainer and model configuration.
        train_dataset (:class:`MIL.Dataset`): Training dataset.
        val_dataset (:class:`MIL.Dataset`): Validation dataset.
        outcomes (str): Outcome column (annotation header) from which to
            derive category labels.
        bags (str): Either a path to directory with \*.pt files, or a list
            of paths to individual \*.pt files. Each file should contain
            exported feature vectors, with each file containing all tile
            features for one patient.

    Keyword args:
        outdir (str): Directory in which to save model and results.
        exp_label (str): Experiment label, used for naming the subdirectory
            in the ``{project root}/mil`` folder, where training history
            and the model will be saved.
        attention_heatmaps (bool): Generate attention heatmaps for slides.
            Not available for multi-modal MIL models. Defaults to False.
        interpolation (str, optional): Interpolation strategy for smoothing
            attention heatmaps. Defaults to 'bicubic'.
        cmap (str, optional): Matplotlib colormap for heatmap. Can be any
            valid matplotlib colormap. Defaults to 'inferno'.
        norm (str, optional): Normalization strategy for assigning heatmap
            values to colors. Either 'two_slope', or any other valid value
            for the ``norm`` argument of ``matplotlib.pyplot.imshow``.
            If 'two_slope', normalizes values less than 0 and greater than 0
            separately. Defaults to None.
        events (str, optional): Annotation column which specifies the
            event, for training a survival model.

    """
    if not isinstance(config, TrainerConfig):
        raise ValueError(f"Unrecognized training configuration of type {type(config)}")

    return config.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        outcomes=outcomes,
        bags=bags,
        outdir=outdir,
        exp_label=exp_label,
        **kwargs
    )

# -----------------------------------------------------------------------------

def build_fastai_learner(
    config: TrainerConfig,
    train_dataset: Dataset,
    val_dataset: Dataset,
    outcomes: Union[str, List[str]],
    bags: Union[str, np.ndarray, List[str]],
    *,
    events: Optional[str] = None,
    outdir: str = 'mil',
    return_shape: bool = False,
    **kwargs
) -> "Learner":
    """Build a FastAI Learner for training an MIL model.

    Does not execute training. Useful for customizing a Learner object
    prior to training.

    Args:
        train_dataset (:class:`MIL.Dataset`): Training dataset.
        val_dataset (:class:`MIL.Dataset`): Validation dataset.
        outcomes (str): Outcome column (annotation header) from which to
            derive category labels.
        bags (str): list of paths to individual \*.pt files. Each file should
            contain exported feature vectors, with each file containing all tile
            features for one patient.
        events (str, optional): Annotation column which specifies the
            event, for training a survival model.

    Keyword args:
        outdir (str): Directory in which to save model and results.
        return_shape (bool): Return the input and output shapes of the model.
            Defaults to False.
        exp_label (str): Experiment label, used for naming the subdirectory
            in the ``outdir`` folder, where training history
            and the model will be saved.
        lr (float): Learning rate, or maximum learning rate if
            ``fit_one_cycle=True``.
        epochs (int): Maximum epochs.
        **kwargs: Additional keyword arguments to pass to the FastAI learner.

    Returns:
        fastai.learner.Learner, and optionally a tuple of input and output shapes
        if ``return_shape=True``.

    """
    from . import _fastai

    labels, unique = utils.get_labels((train_dataset, val_dataset), outcomes, config.model_type, events=events)

    # Prepare bags
    if isinstance(bags, str) or (isinstance(bags, list) and isdir(bags[0])):
        train_bags = train_dataset.get_bags(bags)
        if val_dataset is train_dataset:
            bags = train_bags
        else:
            val_bags = val_dataset.get_bags(bags)
            bags = np.concatenate((train_bags, val_bags))
    else:
        bags = np.array(bags)

    train_slides = train_dataset.slides()
    val_slides = val_dataset.slides()

    if config.aggregation_level == 'slide':
        # Aggregate feature bags across slides.
        bags, targets, train_idx, val_idx = utils.aggregate_trainval_bags_by_slide(
            bags,  # type: ignore
            labels,
            train_slides,
            val_slides,
            log_manifest=(join(outdir, 'slide_manifest.csv') if outdir else None)
        )

    elif config.aggregation_level == 'patient':
        # Associate patients and their slides.
        # This is a dictionary where each key is a slide name and each value
        # is a patient code. Multiple slides can match to the same patient.
        slide_to_patient = { **train_dataset.patients(),
                             **val_dataset.patients() }

        # Aggregate feature bags across patients.
        n_slide_bags = len(bags)
        bags, targets, train_idx, val_idx = utils.aggregate_trainval_bags_by_patient(
            bags,  # type: ignore
            labels,
            train_slides,
            val_slides,
            slide_to_patient=slide_to_patient,
            log_manifest=(join(outdir, 'slide_manifest.csv') if outdir else None)
        )
        log.info(f"Aggregated {n_slide_bags} slide bags to {len(bags)} patient bags.")

    log.info("Training dataset: {} merged bags (from {} possible slides)".format(
        len(train_idx), len(train_slides)))
    log.info("Validation dataset: {} merged bags (from {} possible slides)".format(
        len(val_idx), len(val_slides)))

    # Build FastAI Learner
    learner, (n_in, n_out) = _fastai.build_learner(
        config,
        bags=bags,
        targets=targets,
        train_idx=train_idx,
        val_idx=val_idx,
        unique_categories=unique,
        outdir=outdir,
        **kwargs
    )
    if return_shape:
        return learner, (n_in, n_out)
    else:
        return learner



# Internal training functions.

def _train_multimodal_mixed_mil(
    config: TrainerConfig,
    train_dataset: Dataset,
    val_dataset: Dataset,
    outcomes: Union[str, List[str]],
    bags: Union[str, List[str]],
    *,
    outdir: str = 'mil',
    attention_heatmaps: bool = False,
    uq: bool = False,
    events: Optional[str] = None,
    device: Optional[str] = None,
    **heatmap_kwargs
) -> "Learner":
    """Train an MIL model using FastAI.

    Args:
        train_dataset (:class:`MIL.Dataset`): Training dataset.
        val_dataset (:class:`MIL.Dataset`): Validation dataset.
        outcomes (str): Outcome column (annotation header) from which to
            derive category labels.
        bags (str): Either a path to directory with \*.pt files, or a list
            of paths to individual \*.pt files. Each file should contain
            exported feature vectors, with each file containing all tile
            features for one patient.
        events (str, optional): Annotation column which specifies the
            event, for training a survival model.

    Keyword args:
        outdir (str): Directory in which to save model and results.
        exp_label (str): Experiment label, used for naming the subdirectory
            in the ``{project root}/mil`` folder, where training history
            and the model will be saved.
        lr (float): Learning rate, or maximum learning rate if
            ``fit_one_cycle=True``.
        epochs (int): Maximum epochs.
        attention_heatmaps (bool): Generate attention heatmaps for slides.
            Defaults to False.
        interpolation (str, optional): Interpolation strategy for smoothing
            attention heatmaps. Defaults to 'bicubic'.
        cmap (str, optional): Matplotlib colormap for heatmap. Can be any
            valid matplotlib colormap. Defaults to 'inferno'.
        norm (str, optional): Normalization strategy for assigning heatmap
            values to colors. Either 'two_slope', or any other valid value
            for the ``norm`` argument of ``matplotlib.pyplot.imshow``.
            If 'two_slope', normalizes values less than 0 and greater than 0
            separately. Defaults to None.

    Returns:
        fastai.learner.Learner
    """
    from . import _fastai

    # Prepare validation bags.
    if isinstance(bags, str) or (isinstance(bags, list) and isdir(bags[0])):
        val_bags = val_dataset.get_bags(bags)
    else:
        raise ValueError("Bags must be a path to a directory with .pt files")

    # Build learner.
    learner, (n_in, n_out) = build_fastai_learner(
        config,
        train_dataset,
        val_dataset,
        outcomes,
        events=events,
        bags=bags,
        outdir=outdir,
        device=device,
        return_shape=True
    )

    # Save MIL settings.
    # Attempt to read the unique categories from the learner.
    if not hasattr(learner.dls.train_ds, 'encoder'):
        unique = None
    else:
        encoder = learner.dls.train_ds.encoder
        if encoder is not None:
            unique = encoder.categories_[0].tolist()
        else:
            unique = None
    _log_mil_params(config, outcomes, unique, bags, n_in, n_out, outdir)

    # Train.
    _fastai.train(learner, config)

    # Generate validation predictions.
    df, attention = predict_mil(
        learner.model,
        dataset=val_dataset,
        config=config,
        outcomes=outcomes,
        events=events,
        bags=val_bags,
        attention=True
    )
    df_train, attention_train = predict_mil(
        learner.model,
        dataset=train_dataset,
        config=config,
        outcomes=outcomes,
        events=events,
        bags=bags,
        attention=True
    )
    if outdir:
        pred_out = join(outdir, 'predictions.parquet')
        pred_out_train = join(outdir, 'predictions_train.parquet')
        df.to_parquet(pred_out)
        df_train.to_parquet(pred_out_train)
        log.info(f"Predictions saved to [green]{pred_out}[/]")

    # Print classification metrics, including per-category accuracy
    utils.rename_df_cols(df, outcomes, model_type=config.model_type, inplace=True)
    config.run_metrics(df, level='slide', outdir=outdir)

    # Export attention to numpy arrays
    if attention and outdir:
        utils._export_attention(
            join(outdir, 'attention'),
            attention,
            [path_to_name(b) for b in val_bags]
        )
    return learner

# ------------------------------------------------------------------------------

def _log_mil_params(config, outcomes, unique, bags, n_in, n_out, outdir=None):
    """Log MIL parameters to JSON."""
    mil_params = config.json_dump()
    mil_params['outcomes'] = outcomes
    if unique is not None:
        mil_params['outcome_labels'] = dict(zip(range(len(unique)), unique))
    else:
        mil_params['outcome_labels'] = None
    mil_params['bags'] = bags
    mil_params['input_shape'] = n_in
    mil_params['output_shape'] = n_out
    if isinstance(bags, str) and exists(join(bags, 'bags_config.json')):
        mil_params['bags_extractor'] = load_json(
            join(bags, 'bags_config.json')
        )
    elif isinstance(bags, list):
        mil_params['bags_extractor'] = {}
        for b in bags:
            if isdir(b) and exists(join(b, 'bags_config.json')):
                mil_params['bags_extractor'][b] = load_json(
                    join(b, 'bags_config.json')
                )
            else:
                mil_params['bags_extractor'][b] = None
    else:
        mil_params['bags_extractor'] = None
    if outdir:
        write_json(mil_params, join(outdir, 'mil_params.json'))
    return mil_params