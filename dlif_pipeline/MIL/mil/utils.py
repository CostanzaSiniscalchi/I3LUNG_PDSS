"""Utility functions for MIL."""

import os
import numpy as np
import MIL.util as util
import MIL.errors as errors

from MIL.mil import mil_config, TrainerConfig

from os.path import exists, join, isdir
from typing import Optional, Tuple, Union, Dict, List, Any, TYPE_CHECKING

from MIL.util import log, path_to_name, zip_allowed, load_json, is_tile_size_compatible
from MIL.dataset import Dataset
from MIL.model.torch_utils import get_device

if TYPE_CHECKING:
    import torch


# -----------------------------------------------------------------------------

def load_model_weights(
    weights: str,
    config: Optional[TrainerConfig] = None,
    *,
    input_shape: Optional[int] = None,
    output_shape: Optional[int] = None,
    strict: bool = False
) -> Tuple["torch.nn.Module", TrainerConfig]:
    """Load weights and build model.

    Args:
        weights (str): Path to model weights.
        config (:class:`slideflow.mil.TrainerConfig`):
            Configuration for building model. If ``weights`` is a path to a
            model directory, will attempt to read ``mil_params.json`` from this
            location and load saved configuration. Defaults to None.

    Keyword Args:
        input_shape (int): Number of features in the input data.
        output_shape (int): Number of output classes.
        strict (bool): Whether to strictly enforce that all hyperparameters
            are recognized. Defaults to False.

    Returns:
        :class:`torch.nn.Module`: Loaded model.
    """
    import torch

    if exists(join(weights, 'mil_params.json')):
        mil_params = load_json(join(weights, 'mil_params.json'))
    else:
        mil_params = None

    # Read configuration from saved model, if available
    if config is None:
        if mil_params is None:
            raise errors.ModelError(
                f"Could not find `mil_params.json` at {weights}. Check the "
                "provided model/weights path, or provide a configuration "
                "with 'config'."
            )
        else:
            config = mil_config(
                trainer=mil_params['trainer'],
                **mil_params['params'],
                validate=strict
            )

    # Determine the input and output shapes, reading from the model
    # configuration, if necessary.
    if input_shape is None or output_shape is None:
        if mil_params is None:
            raise errors.ModelError(
                f"Could not find `mil_params.json` at {weights}. Check the "
                "provided model/weights path, or provide the input and output "
                "shape via input_shape and output_shape."
            )
        else:
            if input_shape is None and 'input_shape' in mil_params:
                input_shape = mil_params['input_shape']
            elif input_shape is None:
                raise errors.ModelError(
                    'Could not find input_shape in `mil_params.json`.'
                )
            if output_shape is None and 'output_shape' in mil_params:
                output_shape = mil_params['output_shape']
            elif output_shape is None:
                raise errors.ModelError(
                    'Could not find output_shape in `mil_params.json`.'
                )

    # Build the model
    model = config.build_model(input_shape, output_shape)
    if isdir(weights):
        weights = _find_weights_path(weights, mil_params)
    log.info(f"Loading model weights from [green]{weights}[/]")
    model.load_state_dict(torch.load(weights, map_location=get_device(), weights_only=True))

    # Prepare device.
    if hasattr(model, 'relocate'):
        model.relocate()  # type: ignore
    model.eval()
    return model, config

def aggregate_bags_by_slide(
    bags: np.ndarray,
    labels: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate bags by slide.

    Args:
        bags (np.ndarray): Array of bag paths.
        labels (dict): Dictionary mapping slide names to labels.

    Returns:
        tuple: (bags, targets)

    """
    # Convert bags to nested list, where each sublist contains bags for a slide.
    slides = np.unique([path_to_name(bag) for bag in bags])
    bags = np.array([
        [bag for bag in bags if path_to_name(bag) == slide]
        for slide in slides
    ])

    # Prepare targets, mapping each bag sublist to the label of the first bag.
    targets = np.array([labels[path_to_name(sublist[0])] for sublist in bags])

    return bags, targets

def _find_weights_path(path: str, mil_params: Dict) -> str:
    """Determine location of model weights from a given model directory."""
    if exists(join(path, 'models', 'best_valid.pth')):
        weights = join(path, 'models', 'best_valid.pth')
    elif exists(join(path, 'results', 's_0_checkpoint.pt')):
        weights = join(path, 'results', 's_0_checkpoint.pt')
    elif 'weights' in mil_params and mil_params['weights']:
        if mil_params['weights'].startswith('/'):
            weights = mil_params['weights']
        elif mil_params['weights'].startswith('./'):
            weights = join(path, mil_params['weights'][2:])
        else:
            weights = join(path, mil_params['weights'])
    else:
        raise errors.ModelError(
            f"Could not find model weights at path {path}"
        )
    return weights

def aggregate_bags_by_patient(
    bags: np.ndarray,
    labels: Dict[str, int],
    slide_to_patient: Dict[str, str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate bags by patient.

    Args:
        bags (np.ndarray): Array of bag paths. May be nested, with outer list
            containing sublists of bags for each patient.
        labels (dict): Dictionary mapping slide names to labels.

    Returns:
        tuple: (bags, targets)

    """
    # Create a reverse dictionary, mapping patient codes to a list of bags.
    patient_to_bags = {}  # type: Dict[str, List[str]]
    for bag in bags:
        if not isinstance(bag, str):
            slide_name = path_to_name(bag[0])
        else:
            slide_name = path_to_name(bag)
        patient = slide_to_patient[slide_name]
        if patient not in patient_to_bags:
            patient_to_bags[patient] = []
        patient_to_bags[patient].append(bag)

    # Create array where each element contains the list of slides for a patient.
    bags = np.array([lst for lst in patient_to_bags.values()], dtype=object)

    # Create a dictionary mapping patients to their labels.
    patients_labels = {}
    for patient, patient_bags in patient_to_bags.items():
        # Confirm that all slides for a patient have the same label.
        if len(np.unique([labels[path_to_name((b if isinstance(b, str) else b[0]))] for b in patient_bags])) != 1:
            raise ValueError(
                "Patient {} has slides/bags with different labels".format(patient))
        first_bag = patient_bags[0] if isinstance(patient_bags[0], str) else patient_bags[0][0]
        patients_labels[patient] = labels[path_to_name(first_bag)]

    # Prepare targets, mapping each bag sublist to the label of the first bag.
    targets = np.array([patients_labels[slide_to_patient[path_to_name((sublist[0] if isinstance(sublist[0], str) else sublist[0][0]))]]
                        for sublist in bags])

    return bags, targets


def aggregate_trainval_bags_by_slide(
    bags: np.ndarray,
    labels: Dict[str, int],
    train_slides: List[str],
    val_slides: List[str],
    *,
    log_manifest: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate training/validation bags by slide.

    Args:
        bags (np.ndarray): Array of bag paths.
        labels (dict): Dictionary mapping slide names to labels.
        train_slides (list): List of training slide names.
        val_slides (list): List of validation slide names.

    Keyword Args:
        log_manifest (str): Path to manifest file to write.
            Defaults to None.

    Returns:
        tuple: (bags, targets, train_idx, val_idx)

    """
    # Prepare targets
    targets = np.array([labels[path_to_name(f)] for f in bags])

    # Prepare training/validation indices
    train_idx = np.array([i for i, bag in enumerate(bags)
                        if path_to_name(bag) in train_slides])
    val_idx = np.array([i for i, bag in enumerate(bags)
                        if path_to_name(bag) in val_slides])

    # Write slide/bag manifest
    if log_manifest is not None:
        util.log_manifest(
            [bag for bag in bags if path_to_name(bag) in train_slides],
            [bag for bag in bags if path_to_name(bag) in val_slides],
            labels=labels,
            filename=log_manifest
        )

    return bags, targets, train_idx, val_idx


def _verify_compatible_tile_size(mil_path: str, bag_path: str, strict: bool = False):
    """Verify that the tile size of the MIL model is compatible with the bag.

    Args:
        mil_path: Path to trained model directory, containing mil_params.json.
        bag_path: Path to bag directory, containing bags_config.json



    """
    msg = None
    if not exists(join(mil_path, 'mil_params.json')):
        msg = f"Could not find mil_params.json at {mil_path}; unable to verify tile size compatibility."
    if not exists(join(bag_path, 'bags_config.json')):
        msg = f"Could not find bags_config.json at {bag_path}; unable to verify tile size compatibility."
    if msg and strict:
        raise errors.IncompatibleTileSizeError(msg)
    elif msg:
        log.warning(msg)
        return

    mil_params = load_json(join(mil_path, 'mil_params.json'))
    mil_bags_params = mil_params['bags_extractor']
    bags_config = load_json(join(bag_path, 'bags_config.json'))

    def _has_px(d):
        return 'tile_px' in d and 'tile_um' in d

    # Verify that the slide has the same tile size as the bags
    if _has_px(mil_bags_params) and _has_px(bags_config):
        mil_px, mil_um = mil_bags_params['tile_px'], mil_bags_params['tile_um']
        bag_px, bag_um = bags_config['tile_px'], bags_config['tile_um']
        if not is_tile_size_compatible(bag_px, bag_um, mil_px, mil_um):
            log.error(f"Model tile size (px={mil_px}, um={mil_um}) does not match the tile size "
                      f"of indicated bags (px={bag_px}, um={bag_um}). Predictions may be unreliable.")


def aggregate_trainval_bags_by_patient(
    bags: np.ndarray,
    labels: Dict[str, int],
    train_slides: List[str],
    val_slides: List[str],
    slide_to_patient: Dict[str, str],  # slide -> patient
    *,
    log_manifest: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate training/validation bags by patient.

    Args:
        bags (np.ndarray): Array of bag paths.
        labels (dict): Dictionary mapping slide names to labels.
        train_slides (list): List of training slide names.
        val_slides (list): List of validation slide names.

    Keyword Args:
        log_manifest (str): Path to manifest file to write.
            Defaults to None.

    Returns:
        tuple: (bags, targets, train_idx, val_idx)

    """
    # Create a reverse dictionary, mapping patient codes to a list of bags.
    patient_to_bags = {}  # type: Dict[str, List[str]]
    for bag in bags:
        patient = slide_to_patient[path_to_name(bag)]
        if patient not in patient_to_bags:
            patient_to_bags[patient] = []
        patient_to_bags[patient].append(bag)

    # Create array where each element contains the list of slides for a patient.
    bags = np.array([lst for lst in patient_to_bags.values()], dtype=object)

    # Create a dictionary mapping patients to their labels.
    patients_labels = {}
    for patient, patient_bags in patient_to_bags.items():
        # Confirm that all slides for a patient have the same label.
        if len(np.unique([labels[path_to_name(b)] for b in patient_bags])) != 1:
            raise ValueError(
                "Patient {} has slides/bags with different labels".format(patient))
        patients_labels[patient] = labels[path_to_name(patient_bags[0])]

    # Prepare targets, mapping each bag sublist to the label of the first bag.
    targets = np.array([patients_labels[slide_to_patient[path_to_name(sublist[0])]]
                        for sublist in bags])

    # Identify the bag indices of the training and validation patients.
    train_idx = np.array([i for i, sublist in enumerate(bags)
                            if path_to_name(sublist[0]) in train_slides])
    val_idx = np.array([i for i, sublist in enumerate(bags)
                        if path_to_name(sublist[0]) in val_slides])

    # Write patient/bag manifest
    if log_manifest is not None:
        train_patients = list(np.unique([slide_to_patient[path_to_name(bags[i][0])] for i in train_idx]))
        val_patients = list(np.unique([slide_to_patient[path_to_name(bags[i][0])] for i in val_idx]))
        util.log_manifest(
            train_patients,
            val_patients,
            labels=patients_labels,
            filename=log_manifest,
            remove_extension=False
        )

    return bags, targets, train_idx, val_idx



def get_labels(
    datasets: Union[Dataset, List[Dataset]],
    outcomes: Union[str, List[str]],
    model_type: str,
    *,
    format: str = 'name',
    events: Optional[str] = None,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Get labels for a dataset.

    Args:
        datasets (Dataset or list(Dataset)): Dataset(s) containing labels.
        outcomes (str or list(str)): Outcome(s) to extract.
        model_type (str): Type of model to use.

    Keyword Args:
        format (str): Format for categorical labels. Either 'id' or 'name'.
            Defaults to 'name'.

    """
    if isinstance(datasets, Dataset):
        datasets = [datasets]

    # Prepare labels and slides
    labels = {}
    if model_type in ['classification', 'ordinal', 'multimodal']:
        all_unique = []
        for dts in datasets:
            _labels, _unique = dts.labels(outcomes, format=format)
            labels.update(_labels)
            all_unique.append(_unique)
        unique = np.unique(all_unique)
    elif model_type in ['survival', 'multimodal_survival']:
        if events is None:
            raise ValueError("For survival models, 'events' parameter must be provided")
        for dts in datasets:
            time_labels, _ = dts.labels(outcomes, use_float=True)
            event_labels, _ = dts.labels(events, use_float=True)
            # Create tuples of (time, event) for each slide
            for slide in time_labels:
                labels[slide] = (time_labels[slide][0], event_labels[slide][0])
        unique = None
    else:
        for dts in datasets:
            _labels, _unique = dts.labels(outcomes, use_float=True)
            labels.update(_labels)
        unique = None
    return labels, unique


def rename_df_cols(df, outcomes, model_type, inplace=False):
    """Rename columns of a DataFrame based on outcomes.

    This standarization of column names enables metrics calculation
    to be consistent across different models and outcomes.

    Args:
        df (pd.DataFrame): DataFrame with columns to rename.
            For classification outcomes, there is assumed to be a single "y_true"
            column which will be renamed to "{outcome}-y_true", and multiple
            "y_pred{n}" columns which will be renamed to "{outcome}-y_pred{n}".
            For regression outcomes, there are assumed to be multiple "y_true{n}"
            and "y_pred{n}" columns which will be renamed to "{outcome}-y_true{n}"
            and "{outcome}-y_pred{n}", respectively.
        outcomes (str or list(str)): Outcome(s) to append to column names.
            If there are multiple outcome names, these are joined with a hyphen.
        categorical (bool): Whether the outcomes are categorical.

    """
    if model_type in ['classification', 'ordinal', 'multimodal']:
        return _rename_categorical_df_cols(df, outcomes, inplace=inplace)
    elif model_type in ['survival', 'multimodal_survival']:
        return _rename_survival_df_cols(df, outcomes, inplace=inplace)
    else:
        return _rename_continuous_df_cols(df, outcomes, inplace=inplace)


def _rename_categorical_df_cols(df, outcomes, inplace=False):
    outcome_name = outcomes if isinstance(outcomes, str) else '-'.join(outcomes)
    return df.rename(
        columns={c: f"{outcome_name}-{c}" for c in df.columns if c != 'slide'},
        inplace=inplace
    )


def _rename_continuous_df_cols(df, outcomes, inplace=False):
    if isinstance(outcomes, str):
        outcomes = [outcomes]
    cols_to_rename = {f'y_pred{o}': f"{outcomes[o]}-y_pred" for o in range(len(outcomes))}
    cols_to_rename.update({f'y_true{o}': f"{outcomes[o]}-y_true" for o in range(len(outcomes))})
    return df.rename(columns=cols_to_rename, inplace=inplace)

def _rename_survival_df_cols(df, outcomes, inplace=False):
    df = df.rename(columns={
        'y_true0': 'time-y_true',
        'y_true1': 'event-y_true',
        'y_pred0': 'time-y_pred'
    }, inplace=inplace)
    return df


# -----------------------------------------------------------------------------

def _detect_device(
    model: "torch.nn.Module",
    device: Optional[str] = None,
    verbose: bool = False
) -> "torch.device":
    """Auto-detect device from the given model."""
    import torch

    if device is None:
        device = next(model.parameters()).device
        if verbose:
            log.debug(f"Auto device detection: using {device}")
    elif isinstance(device, str):
        if verbose:
            log.debug(f"Using {device}")
        device = torch.device(device)
    return device


def _export_attention(
    dest: str,
    y_att: Union[List[np.ndarray], List[List[np.ndarray]]],
    slides: List[str]
) -> None:
    """Export attention scores to a directory."""
    if not exists(dest):
        os.makedirs(dest)
    for slide, att in zip(slides, y_att):

        if isinstance(att, (list, tuple)) and not zip_allowed():
            raise RuntimeError(
                "Cannot export multimodal attention scores to a directory (NPZ) "
                "when ZIP functionality is disabled. Enable zip functionality "
                "by setting 'SF_ALLOW_ZIP=1' in your environment, or by "
                "wrapping your script in 'with sf.util.enable_zip():'.")

        elif isinstance(att, (list, tuple)):
            out_path = join(dest, f'{slide}_att.npz')
            np.savez(out_path, *att)

        elif zip_allowed():
            out_path = join(dest, f'{slide}_att.npz')
            np.savez(out_path, att)

        else:
            out_path = join(dest, f'{slide}_att.npy')
            np.save(out_path, att)

    log.info(f"Attention scores exported to [green]{out_path}[/]")

