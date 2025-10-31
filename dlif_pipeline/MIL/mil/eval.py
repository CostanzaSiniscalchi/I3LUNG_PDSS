"""Tools for evaluation MIL models."""

import os
import inspect
import pandas as pd
import numpy as np
import MIL.errors as errors

from rich.progress import track
from os.path import join, exists, dirname
from typing import Union, List, Optional, Callable, Tuple, Any, TYPE_CHECKING
from MIL.util import load_json, write_json, get_new_model_dir, log, path_to_name
from MIL.dataset import Dataset
from ._params import TrainerConfig
from . import utils

if TYPE_CHECKING:
    import torch

# -----------------------------------------------------------------------------
# User-facing API for evaluation and prediction.

def eval_mil(
    weights: str,
    dataset: Dataset,
    outcomes: Union[str, List[str]],
    bags: Union[str, List[str]],
    config: Optional[TrainerConfig] = None,
    *,
    events: Optional[str] = None,
    outdir: str = 'mil',
    attention_heatmaps: bool = False,
    save_attention: bool = False,
    uq: bool = False,
    aggregation_level: Optional[str] = None,
    **heatmap_kwargs
) -> pd.DataFrame:
    """Evaluate a multiple-instance learning model.

    Saves results for the evaluation in the target folder, including
    predictions (parquet format), attention (Numpy format for each slide),
    and attention heatmaps (if ``attention_heatmaps=True``).

    Logs classifier metrics (AUROC and AP) to the console.

    Args:
        weights (str): Path to model weights to load.
        dataset (sf.Dataset): Dataset to evaluation.
        outcomes (str, list(str)): Outcomes.
        bags (str, list(str)): Path to bags, or list of bag file paths.
            Each bag should contain PyTorch array of features from all tiles in
            a slide, with the shape ``(n_tiles, n_features)``.
        config (:class:`slideflow.mil.TrainerConfig`):
            Configuration for building model. If ``weights`` is a path to a
            model directory, will attempt to read ``mil_params.json`` from this
            location and load saved configuration. Defaults to None.

    Keyword arguments:
        outdir (str): Path at which to save results.
        attention_heatmaps (bool): Generate attention heatmaps for slides.
            Not available for multi-modal MIL models. Defaults to False.
        save_attention (bool): Save attention scores as .npz files.
            Defaults to False.
        interpolation (str, optional): Interpolation strategy for smoothing
            attention heatmaps. Defaults to 'bicubic'.
        aggregation_level (str, optional): Aggregation level for predictions.
            Either 'slide' or 'patient'. Defaults to None (uses the model
            configuration).
        cmap (str, optional): Matplotlib colormap for heatmap. Can be any
            valid matplotlib colormap. Defaults to 'inferno'.
        norm (str, optional): Normalization strategy for assigning heatmap
            values to colors. Either 'two_slope', or any other valid value
            for the ``norm`` argument of ``matplotlib.pyplot.imshow``.
            If 'two_slope', normalizes values less than 0 and greater than 0
            separately. Defaults to None.

    """
    # if isinstance(bags, str):
    #     utils._verify_compatible_tile_size(weights, bags)

    model, config = utils.load_model_weights(weights, config)
    model.eval()
    params = {
        'model_path': weights,
        'eval_bags': bags,
        'eval_filters': dataset._filters,
        'mil_params': load_json(join(weights, 'mil_params.json'))
    }
    return config.eval(
        model,
        dataset,
        outcomes,
        bags,
        events=events,
        outdir=outdir,
        attention_heatmaps=attention_heatmaps,
        save_attention=save_attention,
        uq=uq,
        params=params,
        aggregation_level=aggregation_level,
        **heatmap_kwargs
    )


def predict_mil(
    model: Union[str, Callable],
    dataset: "Dataset",
    outcomes: Union[str, List[str]],
    bags: Union[str, np.ndarray, List[str]],
    *,
    events: Optional[str] = None,
    config: Optional[TrainerConfig] = None,
    attention: bool = False,
    aggregation_level: Optional[str] = None,
    **kwargs
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, List[np.ndarray]]]:
    """Generate predictions for a dataset from a saved MIL model.

    Args:
        model (torch.nn.Module): Model from which to generate predictions.
        dataset (sf.Dataset): Dataset from which to generation predictions.
        outcomes (str, list(str)): Outcomes.
        bags (str, list(str)): Path to bags, or list of bag file paths.
            Each bag should contain PyTorch array of features from all tiles in
            a slide, with the shape ``(n_tiles, n_features)``.

    Keyword args:
        config (:class:`slideflow.mil.TrainerConfig`):
            Configuration for the MIL model. Required if model is a loaded ``torch.nn.Module``.
            Defaults to None.
        attention (bool): Whether to calculate attention scores. Defaults to False.
        uq (bool): Whether to generate uncertainty estimates. Experimental. Defaults to False.
        aggregation_level (str): Aggregation level for predictions. Either 'slide'
            or 'patient'. Defaults to None.
        attention_pooling (str): Attention pooling strategy. Either 'avg'
                or 'max'. Defaults to None.

    Returns:
        pd.DataFrame: Dataframe of predictions.

        list(np.ndarray): Attention scores (if ``attention=True``)
    """
    # Load the model
    if isinstance(model, str):
        model_path = model
        model, config = utils.load_model_weights(model_path, config)
        model.eval()

        if isinstance(bags, str):
            utils._verify_compatible_tile_size(model_path, bags)
    elif config is None:
        raise ValueError("If model is not a path, a TrainerConfig object must be provided via the 'config' argument.")

    # Validate aggregation level.
    if aggregation_level is None:
        aggregation_level = config.aggregation_level
    if aggregation_level not in ('slide', 'patient'):
        raise ValueError(
            f"Unrecognized aggregation level: '{aggregation_level}'. "
            "Must be either 'patient' or 'slide'."
        )

    # Prepare labels.
    labels, _ = utils.get_labels(dataset, outcomes, config.model_type, events=events, format='id')

    # Prepare bags and targets.
    slides = list(labels.keys())
    if isinstance(bags, str):
        bags = dataset.get_bags(bags)
    else:
        bags = np.array([b for b in bags if path_to_name(b) in slides])

    # Aggregate bags by slide or patient.
    if aggregation_level == 'patient':
        if config.model_type == 'multimodal_survival':
            raise NotImplementedError(
                "Patient-level aggregation not yet supported for multimodal survival models"
            )
        # Get nested list of bags, aggregated by slide.
        slide_to_patient = dataset.patients()
        n_slide_bags = len(bags)
        bags, y_true = utils.aggregate_bags_by_patient(bags, labels, slide_to_patient)
        log.info(f"Aggregated {n_slide_bags} slide bags to {len(bags)} patient bags.")

        # Create prediction dataframe.
        patients = [slide_to_patient[path_to_name(b[0])] for b in bags]
        df_dict = dict(patient=patients, y_true=y_true)

    else:
        # Ensure slide names are sorted according to the bags.
        slides = [path_to_name(b) for b in bags]
        y_true = np.array([labels[s] for s in slides])

        # Create prediction dataframe.
        df_dict = dict(slide=slides)

        # Handle continous outcomes.
        if len(y_true.shape) > 1:
            for i in range(y_true.shape[-1]):
                df_dict[f'y_true{i}'] = y_true[:, i]
        else:
            df_dict['y_true'] = y_true

    # Inference.
    model.eval()
    pred_out = config.predict(model, bags, attention=attention, **kwargs)
    if kwargs.get('uq'):
        y_pred, y_att, y_uq = pred_out
    else:
        y_pred, y_att = pred_out

    # Update dataframe with predictions.
    for i in range(y_pred.shape[-1]):
        df_dict[f'y_pred{i}'] = y_pred[:, i]
    if kwargs.get('uq'):
        for i in range(y_uq.shape[-1]):
            df_dict[f'uncertainty{i}'] = y_uq[:, i]
    df = pd.DataFrame(df_dict)

    if config.model_type in ['survival', 'multimodal_survival']:
        df['y_pred0'] = -df['y_pred0']

    if attention:
        return df, y_att
    else:
        return df

# -----------------------------------------------------------------------------
# Prediction from bags.

def predict_from_mixed_bags(
    model: "torch.nn.Module",
    bags: List[str],
    *,
    attention: bool = False,
    use_lens: bool = False, # for compatibility with other predict_from_bags
    uq: bool = False, # for compatibility with other predict_from_bags
    device: Optional[Any] = None,
    apply_softmax: Optional[bool] = None,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Generate predictions from mixed multimodal bags.

    Args:
        model: PyTorch model
        bags: List of paths to .pt files containing mixed multimodal features

    Keyword Args:
        attention: Whether to return attention scores
        device: Device on which to run inference
        apply_softmax: Whether to apply softmax to outputs

    Returns:
        Tuple containing:
        - Predictions array (shape: n_bags x n_classes)
        - List of attention arrays (if attention=True)
    """
    import torch
    
    device = utils._detect_device(model, device, verbose=False)
    y_pred = []
    y_att = []

    for bag in bags:
        # Load the multimodal bag dictionary
        bag_dict = torch.load(bag, weights_only=True)  # weights_only=True for torch 2.0+
        
        # Get features and mask
        mask = bag_dict['mask'].to(device)
        features = []
        for i in range(1, len(mask) + 1):
            feat_key = f'feature{i}'
            features.append(bag_dict[feat_key].to(device))

        # Add batch dimension
        features = [f.unsqueeze(0) for f in features]
        mask = mask.unsqueeze(0)

        with torch.inference_mode():
            # Forward pass
            if attention:
                _pred, _att = model(*features, mask, attention=True, decode=False)
                y_att.append(_att.squeeze(0).cpu().numpy())
            else:
                _pred = model(*features, mask)

            if apply_softmax:
                _pred = torch.nn.functional.softmax(_pred, dim=1)
            
            y_pred.append(_pred.cpu().numpy())

    yp = np.concatenate(y_pred, axis=0)
    return yp, y_att if attention else None

def run_eval(
    model: "torch.nn.Module",
    dataset: Dataset,
    outcomes: Union[str, List[str]],
    bags: Union[str, List[str]],
    config: TrainerConfig,
    *,
    events: Optional[str] = None,
    outdir: str = 'mil',
    attention_heatmaps: bool = False,
    save_attention: bool = False,
    uq: bool = False,
    params: Optional[dict] = None,
    aggregation_level: Optional[str] = None,
    **heatmap_kwargs
) -> pd.DataFrame:
    """Evaluate a standard, single-mode multi-instance learning model.

    Args:
        model (torch.nn.Module): Loaded PyTorch MIL model.
        dataset (sf.Dataset): Dataset to evaluation.
        outcomes (str, list(str)): Outcomes.
        bags (str, list(str)): Path to bags, or list of bag file paths.
            Each bag should contain PyTorch array of features from all tiles in
            a slide, with the shape ``(n_tiles, n_features)``.
        config (:class:`slideflow.mil.TrainerConfig`):
            Configuration for building model.

    Keyword arguments:
        outdir (str): Path at which to save results.
        attention_heatmaps (bool): Generate attention heatmaps for slides.
            Defaults to False.
        save_attention (bool): Save attention scores as .npz files.
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
        pd.DataFrame: Dataframe of predictions.
    """
    # Generate predictions.
    predict_kwargs = dict(
        model=model,
        dataset=dataset,
        config=config,
        outcomes=outcomes,
        events=events,
        bags=bags,
        attention=True,
        aggregation_level=aggregation_level
    )
    if config.is_multimodal:
        if uq:
            log.warning("Uncertainty estimates are not supported for multi-modal models.")
        df, y_att = predict_multimodal_mil(**predict_kwargs)
    else:
        df, y_att = predict_mil(uq=uq, **predict_kwargs)

    # Save results.
    if outdir:
        if not exists(outdir):
            os.makedirs(outdir)
        model_dir = get_new_model_dir(outdir, config.model_config.model)
        if params is not None:
            write_json(params, join(model_dir, 'mil_params.json'))
        pred_out = join(model_dir, 'predictions.parquet')
        df.to_parquet(pred_out)
        log.info(f"Predictions saved to [green]{pred_out}[/]")
    else:
        model_dir = None

    # Print classification metrics, including per-category accuracy)
    metrics_df = utils.rename_df_cols(df, outcomes, model_type=config.model_type)
    config.run_metrics(metrics_df, level='slide', outdir=model_dir)

    # Export attention
    if outdir and y_att and save_attention:
        if 'slide' in df.columns:
            slides_or_patients = df.slide.values
        elif 'patient' in df.columns:
            slides_or_patients = df.patient.values
        else:
            raise ValueError("Malformed dataframe; cannot find 'slide' or 'patient' column.")
        utils._export_attention(join(model_dir, 'attention'), y_att, slides_or_patients)

    # # Attention heatmaps
    # # Not supported for multimodal models
    # if attention_heatmaps and not config.is_multimodal:
    #     log.warning("Cannot generate attention heatmaps for multi-modal models.")
    # elif outdir and y_att and attention_heatmaps:
    #     generate_attention_heatmaps(
    #         outdir=join(model_dir, 'heatmaps'),
    #         dataset=dataset,
    #         bags=bags,  # type: ignore
    #         attention=y_att,
    #         **heatmap_kwargs
    #     )

    return df

def predict_multimodal_mil(
    model: Union[str, Callable],
    dataset: "Dataset",
    outcomes: Union[str, List[str]],
    bags: Union[np.ndarray, List[List[str]]],
    *,
    config: Optional[TrainerConfig] = None,
    attention: bool = False,
    aggregation_level: Optional[str] = None,
    **kwargs
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, List[np.ndarray]]]:
    """Generate predictions for a dataset from a saved multimodal MIL model.

    Args:
        model (torch.nn.Module): Model from which to generate predictions.
        dataset (sf.Dataset): Dataset from which to generation predictions.
        outcomes (str, list(str)): Outcomes.
        bags (str, list(str)): Path to bags, or list of bag file paths.
            Each bag should contain PyTorch array of features from all tiles in
            a slide, with the shape ``(n_tiles, n_features)``.

    Keyword args:
        config (:class:`slideflow.mil.TrainerConfig`):
            Configuration for the MIL model. Required if model is a loaded ``torch.nn.Module``.
            Defaults to None.
        attention (bool): Whether to calculate attention scores. Defaults to False.
        uq (bool): Whether to generate uncertainty estimates. Defaults to False.
        aggregation_level (str): Aggregation level for predictions. Either 'slide'
            or 'patient'. Defaults to None.
        attention_pooling (str): Attention pooling strategy. Either 'avg'
                or 'max'. Defaults to None.

    Returns:
        pd.DataFrame: Dataframe of predictions.

        list(np.ndarray): Attention scores (if ``attention=True``)
    """

    # Load the model
    if isinstance(model, str):
        model_path = model
        model, config = utils.load_model_weights(model_path, config)
        model.eval()

        # Verify tile size compatibility for each bag source.
        for b in bags:
            if isinstance(b, str):
                utils._verify_compatible_tile_size(model_path, b)
    elif config is None:
        raise ValueError("If model is not a path, a TrainerConfig object must be provided via the 'config' argument.")

    # Validate aggregation level.
    if aggregation_level is not None and aggregation_level != 'slide':
        raise ValueError(
            f"Unrecognized aggregation level: '{aggregation_level}'. "
            "Multimodal MIL models only support 'slide' aggregation."
        )

    # Prepare labels.
    labels, _ = utils.get_labels(dataset, outcomes, config.model_type, format='id')

    # Prepare bags and targets.
    slides = list(labels.keys())

    # Load multimodal bags.
    if isinstance(bags[0], str):
        bags, val_slides = utils._get_nested_bags(dataset, bags)

    # This is where we would aggregate bags by slide or patient.
    # This is not yet supported.

    # Ensure slide names are sorted according to the bags.
    slides = [path_to_name(b[0]) for b in bags]
    y_true = np.array([labels[s] for s in slides])

    # Create prediction dataframe.
    df_dict = dict(slide=slides)

    # Handle continous outcomes.
    if len(y_true.shape) > 1:
        for i in range(y_true.shape[-1]):
            df_dict[f'y_true{i}'] = y_true[:, i]
    else:
        df_dict['y_true'] = y_true

    # Inference.
    model.eval()
    y_pred, y_att = config.predict(model, bags, attention=attention, **kwargs)

    # Update dataframe with predictions.
    for i in range(y_pred.shape[-1]):
        df_dict[f'y_pred{i}'] = y_pred[:, i]
    df = pd.DataFrame(df_dict)

    if attention:
        return df, y_att
    else:
        return df
