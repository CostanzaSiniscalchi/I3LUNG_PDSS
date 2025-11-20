import atexit
import csv
import importlib.util
import json
import logging
import os
import re
import torch
import shutil
import sys
import pandas as pd
import threading
import multiprocessing as mp
import time
from rich import progress
from rich.logging import RichHandler
from rich.highlighter import NullHighlighter
from rich.progress import Progress, TextColumn, BarColumn, SpinnerColumn, TextColumn
from contextlib import contextmanager
from glob import glob
from os.path import dirname, exists, isdir, join
from typing import (
    Any, Dict, List, Optional, Tuple, Union, Iterator
)
import numpy as np
from MIL import errors
from . import log_utils

tf_available = importlib.util.find_spec('tensorflow')
torch_available = importlib.util.find_spec('torch')

# Enable color sequences on Windows
try:
    import ctypes
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
except Exception:
    pass


# --- Global vars -------------------------------------------------------------

SUPPORTED_FORMATS = ['svs', 'tif', 'ndpi', 'vms', 'vmu', 'scn', 'mrxs',
                     'tiff', 'svslide', 'bif', 'jpg', 'jpeg', 'png',
                     'ome.tif', 'ome.tiff']
EMPTY = ['', ' ', None, np.nan]
CPLEX_AVAILABLE = (importlib.util.find_spec('cplex') is not None)
try:
    # import pyomo.environ as pyo
    from pyomo.opt import SolverFactory
    opt = SolverFactory('bonmin', validate=False)
    if not opt.available():
        raise errors.SolverNotFoundError
except Exception:
    BONMIN_AVAILABLE = False
else:
    BONMIN_AVAILABLE = True


# --- Commonly used types -----------------------------------------------------

# Outcome labels
Labels = Union[Dict[str, str], Dict[str, int], Dict[str, List[float]]]

# Normalizer fit keyword arguments
NormFit = Union[Dict[str, np.ndarray], Dict[str, List]]

# --- Configure logging--------------------------------------------------------

log = logging.getLogger('MIL')
log.setLevel(logging.DEBUG)

def _shortname(string: str):
    if len(string) == 60:
        # May be TCGA slide with long name; convert to
        # patient name by returning first 12 characters
        return string[:12]
    else:
        return string
    
def getLoggingLevel():
    """Return the current logging level."""
    return log.handlers[0].level


@contextmanager

def addLoggingFileHandler(path):
    fh = logging.FileHandler(path)
    fh.setFormatter(log_utils.FileFormatter())
    handler = log_utils.MultiProcessingHandler(
        "mp-file-handler-{0}".format(len(log.handlers)),
        sub_handler=fh
    )
    log.addHandler(handler)
    atexit.register(handler.close)


# Add tqdm-friendly stream handler
#ch = log_utils.TqdmLoggingHandler()
ch = RichHandler(
    markup=True,
    log_time_format="[%X]",
    show_path=False,
    highlighter=NullHighlighter(),
    rich_tracebacks=True
)
ch.setFormatter(log_utils.LogFormatter())
if 'SF_LOGGING_LEVEL' in os.environ:
    try:
        intLevel = int(os.environ['SF_LOGGING_LEVEL'])
        ch.setLevel(intLevel)
    except ValueError:
        pass
else:
    ch.setLevel(logging.INFO)
log.addHandler(ch)

# Add multiprocessing-friendly file handler
try:
    addLoggingFileHandler("MIL.log")
except Exception as e:
    # If we can't write to the log file, just ignore it
    pass

# Workaround for duplicate logging with TF 2.9
log.propagate = False


class TileExtractionSpeedColumn(progress.ProgressColumn):
    """Renders human readable transfer speed."""

    def render(self, task: "progress.Task") -> progress.Text:
        """Show data transfer speed."""
        speed = task.finished_speed or task.speed
        if speed is None:
            return progress.Text("?", style="progress.data.speed")
        data_speed = f'{int(speed)} img'
        return progress.Text(f"{data_speed}/s", style="progress.data.speed")


class LabeledMofNCompleteColumn(progress.MofNCompleteColumn):
    """Renders a completion column with labels."""

    def __init__(self, unit: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unit = unit

    def render(self, task: "progress.Task") -> progress.Text:
        """Show completion status with labels."""
        if task.total is None:
            return progress.Text("?", style="progress.spinner")
        return progress.Text(
            f"{task.completed}/{task.total} {self.unit}",
            style="progress.spinner"
        )


class ImgBatchSpeedColumn(progress.ProgressColumn):
    """Renders human readable transfer speed."""

    def __init__(self, batch_size=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size

    def render(self, task: "progress.Task") -> progress.Text:
        """Show data transfer speed."""
        speed = task.finished_speed or task.speed
        if speed is None:
            return progress.Text("?", style="progress.data.speed")
        data_speed = f'{int(speed * self.batch_size)} img'
        return progress.Text(f"{data_speed}/s", style="progress.data.speed")


class TileExtractionProgress(Progress):
    def get_renderables(self):
        for task in self.tasks:
            if task.fields.get("progress_type") == 'speed':
                self.columns = (
                    TextColumn("[progress.description]{task.description}"),
                    TileExtractionSpeedColumn()
                )
            if task.fields.get("progress_type") == 'slide_progress':
                self.columns = (
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    progress.TaskProgressColumn(),
                    progress.MofNCompleteColumn(),
                    "●",
                    progress.TimeRemainingColumn(),
                )
            yield self.make_tasks_table([task])


class FeatureExtractionProgress(Progress):
    def get_renderables(self):
        for task in self.tasks:
            if task.fields.get("progress_type") == 'speed':
                self.columns = (
                    TextColumn("[progress.description]{task.description}"),
                    TileExtractionSpeedColumn(),
                    LabeledMofNCompleteColumn('tiles'),
                    "●",
                    progress.TimeRemainingColumn(),
                )
            if task.fields.get("progress_type") == 'slide_progress':
                self.columns = (
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    progress.TaskProgressColumn(),
                    LabeledMofNCompleteColumn('slides')
                )
            yield self.make_tasks_table([task])


class MultiprocessProgressTracker:
    """Wrapper for a rich.progress tracker that can be shared across processes."""

    def __init__(self, tasks):
        ctx = mp.get_context('spawn')
        self.mp_values = {
            task.id: ctx.Value('i', task.completed)
            for task in tasks
        }

    def advance(self, id, amount):
        with self.mp_values[id].get_lock():
            self.mp_values[id].value += amount

    def __getitem__(self, id):
        return self.mp_values[id].value

class MultiprocessProgress:
    """Wrapper for a rich.progress bar that can be shared across processes."""

    def __init__(self, pb):
        self.pb = pb
        self.tracker = MultiprocessProgressTracker(self.pb.tasks)
        self.should_stop = False

    def _update_progress(self):
        while not self.should_stop:
            for task in self.pb.tasks:
                self.pb.update(task.id, completed=self.tracker[task.id])
            time.sleep(0.1)

    def __enter__(self):
        self._thread = threading.Thread(target=self._update_progress)
        self._thread.start()
        return self

    def __exit__(self, *args):
        self.should_stop = True
        self._thread.join()



# --- Utility functions and classes -------------------------------------------

class no_scope():
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access
    with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

def zip_allowed() -> bool:
    return not ('SF_ALLOW_ZIP' in os.environ and os.environ['SF_ALLOW_ZIP'] == '0')

@contextmanager
def enable_zip(enable: bool) -> Iterator[None]:
    _zip_allowed = zip_allowed()
    os.environ['SF_ALLOW_ZIP'] = '1' if enable else '0'
    yield
    os.environ['SF_ALLOW_ZIP'] = '0' if not _zip_allowed else '1'


def as_list(arg1: Any) -> List[Any]:
    if not isinstance(arg1, list):
        return [arg1]
    else:
        return arg1


def is_project(path: str) -> bool:
    """Checks if the given path is a valid MIL project."""
    return isdir(path) and exists(join(path, 'settings.json'))


    """Checks if the given model path points to a UQ-enabled model."""
    is_model_path = (is_tensorflow_model_path(model_path)
                     or is_torch_model_path(model_path))
    if not is_model_path:
        return False
    config = get_model_config(model_path)
    return config['hp']['uq']

def is_slide(path: str) -> bool:
    """Checks if the given path is a supported slide."""
    return (os.path.isfile(path)
            and path_to_ext(path).lower() in SUPPORTED_FORMATS)

def get_slide_paths(slides_dir: str) -> List[str]:
    '''Get all slide paths from a given directory containing slides.'''
    slide_list = [i for i in glob(join(slides_dir, '**/*.*')) if is_slide(i)]
    slide_list.extend([i for i in glob(join(slides_dir, '*.*')) if is_slide(i)])
    return slide_list

def make_dir(_dir: str) -> None:
    """Makes a directory if one does not already exist,
    in a manner compatible with multithreading.
    """
    if not exists(_dir):
        try:
            os.makedirs(_dir, exist_ok=True)
        except FileExistsError:
            pass


def relative_path(path: str, root: str):
    """Returns a relative path, from a given root directory."""
    if path[0] == '.':
        return join(root, path[2:])
    elif path.startswith('$ROOT'):
        raise ValueError("Invalid path prefix $ROOT; update project settings")
    else:
        return path


def global_path(root: str, path_string: str):
    '''Returns global path from a local path.'''
    if not root:
        root = ""
    if path_string and (len(path_string) > 2) and path_string[:2] == "./":
        return os.path.join(root, path_string[2:])
    elif path_string and (path_string[0] != "/"):
        return os.path.join(root, path_string)
    else:
        return path_string


def yes_no_input(prompt: str, default: str = 'no') -> bool:
    '''Prompts user for yes/no input.'''
    while True:
        response = input(prompt)
        if not response and default:
            return (default in ('yes', 'y'))
        elif response.lower() in ('yes', 'no', 'y', 'n'):
            return (response.lower() in ('yes', 'y'))
        else:
            print("Invalid response.")


def path_input(
    prompt: str,
    root: str,
    default: Optional[str] = None,
    create_on_invalid: bool = False,
    filetype: Optional[str] = None,
    verify: bool = True
) -> str:
    '''Prompts user for directory input.'''
    while True:
        relative_response = input(f"{prompt}")
        reponse = global_path(root, relative_response)
        if not relative_response and default:
            relative_response = default
            reponse = global_path(root, relative_response)
        if verify and not os.path.exists(reponse):
            if not filetype and create_on_invalid:
                prompt = f'Path "{reponse}" does not exist. Create? [Y/n] '
                if yes_no_input(prompt, default='yes'):
                    os.makedirs(reponse)
                    return relative_response
                else:
                    continue
            elif filetype:
                print(f'Unable to locate file "{reponse}"')
                continue
        elif not filetype and not os.path.exists(reponse):
            print(f'Unable to locate directory "{reponse}"')
            continue
        resp_type = path_to_ext(reponse)
        if filetype and (resp_type != filetype):
            print(f'Incorrect filetype "{resp_type}", expected "{filetype}"')
            continue
        return relative_response


def choice_input(prompt, valid_choices, default=None, multi_choice=False,
                 input_type=str):
    '''Prompts user for multi-choice input.'''
    while True:
        response = input(f"{prompt}")
        if not response and default:
            return default
        if not multi_choice and response not in valid_choices:
            print("Invalid option.")
            continue
        elif multi_choice:
            try:
                replaced = response.replace(" ", "")
                response = [input_type(r) for r in replaced.split(',')]
            except ValueError:
                print(f"Invalid selection (response: {response})")
                continue
            invalid = [r not in valid_choices for r in response]
            if any(invalid):
                print(f'Invalid selection (response: {response})')
                continue
        return response


def load_json(filename: str) -> Any:
    '''Reads JSON data from file.'''
    with open(filename, 'r') as data_file:
        return json.load(data_file)


class ValidJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            return super().default(obj)
        except TypeError:
            return "<unknown>"


def write_json(data: Any, filename: str) -> None:
    """Write data to JSON file.

    Args:
        data (Any): Data to write.
        filename (str): Path to JSON file.

    """
    # First, remove any invalid entries that are not serializable
    with open(filename, "w") as data_file:
        json.dump(data, data_file, indent=1, cls=ValidJSONEncoder)


def log_manifest(
    train_tfrecords: Optional[List[str]] = None,
    val_tfrecords: Optional[List[str]] = None,
    *,
    labels: Optional[Dict[str, Any]] = None,
    filename: Optional[str] = None,
    remove_extension: bool = True
) -> str:
    """Saves the training manifest in CSV format and returns as a string.

    Args:
        train_tfrecords (list(str], optional): List of training TFRecords.
            Defaults to None.
        val_tfrecords (list(str], optional): List of validation TFRecords.
            Defaults to None.

    Keyword args:
        labels (dict, optional): TFRecord outcome labels. Defaults to None.
        filename (str, optional): Path to CSV file to save. Defaults to None.
        remove_extension (bool, optional): Remove file extension from slide
            names. Defaults to True.

    Returns:
        str: Saved manifest in str format.
    """
    out = ''
    has_labels = (isinstance(labels, dict) and len(labels))
    if filename:
        save_file = open(os.path.join(filename), 'w')
        writer = csv.writer(save_file)
        writer.writerow(['Subject', 'dataset', 'outcome_label'])
    if train_tfrecords or val_tfrecords:
        if train_tfrecords:
            for tfrecord in train_tfrecords:
                if remove_extension:
                    slide = path_to_name(tfrecord)
                else:
                    slide = tfrecord
                outcome_label = labels[slide] if has_labels else 'NA'
                out += ' '.join([slide, 'training', str(outcome_label)])
                if filename:
                    writer.writerow([slide, 'training', outcome_label])
        if val_tfrecords:
            for tfrecord in val_tfrecords:
                if remove_extension:
                    slide = path_to_name(tfrecord)
                else:
                    slide = tfrecord
                outcome_label = labels[slide] if has_labels else 'NA'
                out += ' '.join([slide, 'validation', str(outcome_label)])
                if filename:
                    writer.writerow([slide, 'validation', outcome_label])
    if filename:
        save_file.close()
    return out



    '''Get all slide paths from a given directory containing slides.'''
    slide_list = [i for i in glob(join(slides_dir, '**/*.*')) if is_slide(i)]
    slide_list.extend([i for i in glob(join(slides_dir, '*.*')) if is_slide(i)])
    return slide_list


def read_annotations(path: str) -> Tuple[List[str], List[Dict]]:
    '''Read an annotations file.'''
    results = []
    with open(path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        # First, try to open file
        try:
            header = next(csv_reader, None)
        except OSError:
            raise OSError(
                f"Failed to open annotations file {path}"
            )
        assert isinstance(header, list)
        for row in csv_reader:
            row_dict = {}
            for i, key in enumerate(header):
                row_dict[key] = row[i]
            results += [row_dict]
    return header, results


def path_to_name(path: str) -> str:
    '''Returns name of a file, without extension,
    from a given full path string.'''
    _file = os.path.basename(path)
    dot_split = _file.split('.')
    if len(dot_split) == 1:
        return _file
    elif len(dot_split) > 2 and '.'.join(dot_split[-2:]) in SUPPORTED_FORMATS:
        return '.'.join(dot_split[:-2])
    else:
        return '.'.join(dot_split[:-1])


def path_to_ext(path: str) -> str:
    '''Returns extension of a file path string.'''
    _file = os.path.basename(path)
    dot_split = _file.split('.')
    if len(dot_split) == 1:
        return ''
    elif len(dot_split) > 2 and '.'.join(dot_split[-2:]) in SUPPORTED_FORMATS:
        return '.'.join(dot_split[-2:])
    else:
        return dot_split[-1]


def update_results_log(
    results_log_path: str,
    model_name: str,
    results_dict: Dict
) -> None:
    '''Dynamically update results_log when recording training metrics.'''
    # First, read current results log into a dictionary
    results_log = {}  # type: Dict[str, Any]
    if exists(results_log_path):
        with open(results_log_path, "r") as results_file:
            reader = csv.reader(results_file)
            try:
                headers = next(reader)
            except StopIteration:
                pass
            else:
                try:
                    model_name_i = headers.index('model_name')
                    result_keys = [k for k in headers if k != 'model_name']
                except ValueError:
                    model_name_i = headers.index('epoch')
                    result_keys = [k for k in headers if k != 'epoch']
                for row in reader:
                    name = row[model_name_i]
                    results_log[name] = {}
                    for result_key in result_keys:
                        result = row[headers.index(result_key)]
                        results_log[name][result_key] = result
        # Move the current log file into a temporary file
        shutil.move(results_log_path, f"{results_log_path}.temp")

    # Next, update the results log with the new results data
    for epoch in results_dict:
        results_log.update({f'{model_name}-{epoch}': results_dict[epoch]})

    # Finally, create a new log file incorporating the new data
    with open(results_log_path, "w") as results_file:
        writer = csv.writer(results_file)
        result_keys = []
        # Search through results to find all results keys
        for model in results_log:
            result_keys += list(results_log[model].keys())
        # Remove duplicate result keys
        result_keys = list(set(result_keys))
        result_keys.sort()
        # Write header labels
        writer.writerow(['model_name'] + result_keys)
        # Iterate through model results and record
        for model in results_log:
            row = [model]
            # Include all saved metrics
            for result_key in result_keys:
                if result_key in results_log[model]:
                    row += [results_log[model][result_key]]
                else:
                    row += [""]
            writer.writerow(row)

    # Delete the old results log file
    if exists(f"{results_log_path}.temp"):
        os.remove(f"{results_log_path}.temp")


def tile_size_label(tile_px: int, tile_um: Union[str, int]) -> str:
    """Return the string label of the given tile size."""
    if isinstance(tile_um, str):
        return f"{tile_px}px_{tile_um.lower()}"
    else:
        return f"{tile_px}px_{tile_um}um"


def get_valid_model_dir(root: str) -> List:
    '''
    This function returns the path of the first indented directory from root.
    This only works when the indented folder name starts with a 5 digit number,
    like "00000%".

    Examples
        If the root has 3 files:
        root/00000-foldername/
        root/00001-foldername/
        root/00002-foldername/

        The function returns "root/00000-foldername/"
    '''

    prev_run_dirs = [
        x for x in os.listdir(root)
        if isdir(join(root, x))
    ]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    return prev_run_ids, prev_run_dirs


def get_new_model_dir(root: str, model_name: str) -> str:
    prev_run_ids, prev_run_dirs = get_valid_model_dir(root)
    cur_id = max(prev_run_ids, default=-1) + 1
    model_dir = os.path.join(root, f'{cur_id:05d}-{model_name}')
    assert not os.path.exists(model_dir)
    os.makedirs(model_dir)
    return model_dir


def create_new_model_dir(root: str, model_name: str) -> str:
    """
    If model_name is an absolute path, create it directly and return it.
    Otherwise, generate a new numbered folder in `root` with the name. ludo
    """
    if not os.path.isabs(model_name):
        path = get_new_model_dir(root, model_name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


# --- TFRecord utility functions ----------------------------------------------

    """
    Tessellate a complex polygon, possibly with holes.

    :param vertices: A list of vertices [(x1, y1), (x2, y2), ...] defining the polygon boundary.
    :param holes: An optional list of points [(hx1, hy1), (hx2, hy2), ...] inside each hole in the polygon.
    :return: A numpy array of vertices for the tessellated triangles.
    """
    import triangle as tr

    # Prepare the segment information for the exterior boundary
    segments = np.array([[i, (i + 1) % len(vertices)] for i in range(len(vertices))])

    # Prepare the polygon for Triangle
    polygon = {'vertices': np.array(vertices), 'segments': segments}

    # If there are holes and hole boundaries, add them to the polygon definition
    if hole_points is not None and hole_vertices is not None and len(hole_vertices):
        polygon['holes'] = np.array(hole_points).astype(np.float32)

        # Start adding hole segments after the exterior segments
        start_idx = len(vertices)
        for hole in hole_vertices:
            hole_segments = [[start_idx + i, start_idx + (i + 1) % len(hole)] for i in range(len(hole))]
            segments = np.vstack([segments, hole_segments])
            start_idx += len(hole)

        # Update the vertices and segments in the polygon
        all_vertices = np.vstack([vertices] + hole_vertices)
        polygon['vertices'] = all_vertices
        polygon['segments'] = segments

    # Tessellate the polygon
    tess = tr.triangulate(polygon, 'pF')

    # Extract tessellated triangle vertices
    if 'triangles' not in tess:
        return None

    tessellated_vertices = np.array([tess['vertices'][t] for t in tess['triangles']]).reshape(-1, 2)

    # Convert to float32
    tessellated_vertices = tessellated_vertices.astype('float32')

    return tessellated_vertices

def is_mag(arg1: str) -> bool:
    arg1_split = arg1.lower().split('x')
    if (len(arg1_split) != 2) or (arg1_split[1] != ''):
        return False
    try:
        mag = float(arg1_split[0])
    except ValueError:
        return False
    return True

def assert_is_mag(arg1: str):
    if not isinstance(arg1, str) or not is_mag(arg1):
        raise ValueError(
            f'Invalid magnification {arg1}. Must be of format'
            f' [int/float]x, such as "10x", "20X", or "2.5x"'
        )

def to_mag(arg1: str) -> Union[int, float]:
    assert_is_mag(arg1)
    try:
        return int(arg1.lower().split('x')[0])
    except ValueError:
        return float(arg1.lower().split('x')[0])

def is_tile_size_compatible(
    tile_px1: int,
    tile_um1: Union[str, int],
    tile_px2: int,
    tile_um2: Union[str, int]
) -> bool:
    """Check whether tile sizes are compatible.

    Compatibility is defined as:
        - Equal size in pixels
        - If tile width (tile_um) is defined in microns (int) for both, these must be equal
        - If tile width (tile_um) is defined as a magnification (str) for both, these must be equal
        - If one is defined in microns and the other as a magnification, the calculated magnification must be +/- 2.

    Example 1:
    - tile_px1=299, tile_um1=302
    - tile_px2=299, tile_um2=304
    - Incompatible (unequal micron width)

    Example 2:
    - tile_px1=299, tile_um1=10x
    - tile_px2=299, tile_um2=9x
    - Incompatible (unequal magnification)

    Example 3:
    - tile_px1=299, tile_um1=302
    - tile_px2=299, tile_um2=10x
    - Compatible (first has an equivalent magnification of 9.9x, which is +/- 2 compared to 10x)


    Args:
        tile_px1 (int): Tile size (in pixels) of first slide.
        tile_um1 (int or str): Tile size (in microns) of first slide.
            Can also be expressed as a magnification level, e.g. ``'10x'``
        tile_px2 (int): Tile size (in pixels) of second slide.
        tile_um2 (int or str): Tile size (in microns) of second slide.
            Can also be expressed as a magnification level, e.g. ``'10x'``

    Returns:
        bool: Whether the tile sizes are compatible.

    """
    # Type checks
    if not isinstance(tile_px1, int):
        raise ValueError("Expected tile_px1 to be an int, got: {}".format(type(tile_px1)))
    if not isinstance(tile_um1, (str, int)):
        raise ValueError("Expected tile_um1 to be a str or int, got: {}".format(type(tile_um1)))
    if not isinstance(tile_px2, int):
        raise ValueError("Expected tile_px2 to be an int, got: {}".format(type(tile_px2)))
    if not isinstance(tile_um2, (str, int)):
        raise ValueError("Expected tile_um2 to be a str or int, got: {}".format(type(tile_um2)))

    # Enforce equivalent pixel size
    if tile_px1 != tile_px2:
        return False
    # If both are defined as a magnification, check if these are equal
    if isinstance(tile_um1, str) and isinstance(tile_um2, str):
        return tile_um1 == tile_um2
    # If both are defined in microns, check if these are equal
    if isinstance(tile_um1, int) and isinstance(tile_um2, int):
        return tile_um1 == tile_um2
    # If one is defined in microns and the other as magnification,
    # check if they are compatible.
    if isinstance(tile_um1, str) and isinstance(tile_um2, int):
        mag2 = 10 / (tile_um2 / tile_px2)
        return abs(mag2 - to_mag(tile_um1)) <= 2
    if isinstance(tile_um1, int) and isinstance(tile_um2, str):
        mag1 = 10 / (tile_um1 / tile_px1)
        return abs(mag1 - to_mag(tile_um2)) <= 2
    else:
        raise ValueError("Error assessing tile size compatibility between px={}, um={} and px={}, um={}".format(
            tile_px1, tile_um1, tile_px2, tile_um2
        ))


def prepare_multimodal_mixed_bags(path: str, bags_path: str) -> None:
    """Prepare multimodal mixed bags from a dataframe file.

    Processes a dataframe containing multimodal features where some modalities may be
    missing (represented as None/NaN) for certain samples. Creates one .pt file per slide
    containing features and a mask indicating present/missing modalities.

    Args:
        path (str): Path to dataframe file (.csv, .parquet, or .xlsx). The dataframe
            must have:
            - A 'subject' column containing unique identifiers
            - Feature columns containing arrays/lists or None/NaN values

    Example structure of saved .pt files:
        {
            'feature1': torch.tensor([...]),  # Original features or zeros if missing
            'feature2': torch.tensor([...]),  # Original features or zeros if missing
            'feature3': torch.tensor([...]),  # Original features or zeros if missing
            'mask': torch.tensor([True, False, True])  # Boolean vector showing present features
        }
    """

    # Read the dataframe based on file extension
    try:
        if path.endswith('.csv'):
            df = pd.read_csv(path)
        elif path.endswith('.parquet') or path.endswith('.pt'):
            df = pd.read_parquet(path)
        elif path.endswith('.xlsx'):
            df = pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported file extension in {path}")
    except Exception as e:
        raise ValueError(f"Error reading file {path}: {str(e)}")

    # Verify subject column exists
    if 'Subject' not in df.columns:
        raise ValueError("DataFrame must contain 'Subject' column")

    # Get feature columns (all except 'subject') and rename them
    feature_cols = [col for col in df.columns if col != 'Subject']
    if not feature_cols:
        raise ValueError("No feature columns found in DataFrame")
    
    # Create mapping of original column names to feature1, feature2, etc.
    feature_map = {col: f'feature{i+1}' for i, col in enumerate(feature_cols)}

    # Create output directory
    base_dir = dirname(path) if dirname(path) else os.getcwd()
    outdir = join(base_dir, bags_path)
    os.makedirs(outdir, exist_ok=True)

    # First pass: determine feature dimensions for each modality
    log.info("Determining feature dimensions...")
    feature_dims = {}
    for col in feature_cols:
        for value in df[col]:
            if not (value is None or (isinstance(value, (float, str)) and pd.isna(value))):
                if isinstance(value, (list, np.ndarray)):
                    feature_dims[col] = len(value)
                    break

    # Process each slide with progress bar
    slides = df['Subject'].unique()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TextColumn("[progress.remaining]{task.completed}/{task.total}"),
    ) as progress:
        task = progress.add_task("Processing slides...", total=len(slides))
        
        for slide in slides:
            slide_data = df[df['Subject'] == slide].iloc[0]
            
            # Initialize dictionary for this slide
            slide_dict = {}
            
            # Create mask showing which features are present
            mask = []
            
            # Process each feature
            for orig_feat, new_feat in feature_map.items():
                value = slide_data[orig_feat]
                # Check if feature is present (not None/NaN)
                if value is None or (isinstance(value, (float, str)) and pd.isna(value)):
                    # Create zero tensor of appropriate size
                    slide_dict[new_feat] = torch.zeros(feature_dims[orig_feat], dtype=torch.float32).reshape(-1)
                    mask.append(False)
                else:
                    # Convert to tensor if not already
                    if isinstance(value, (list, np.ndarray)):
                        slide_dict[new_feat] = torch.tensor(value, dtype=torch.float32)
                    else:
                        slide_dict[new_feat] = value
                    mask.append(True)
            
            # Add boolean mask to dictionary
            slide_dict['mask'] = torch.tensor(mask, dtype=torch.bool)
            
            # Save to .pt file
            torch.save(slide_dict, join(outdir, f"{slide}.pt"))
            
            # Update progress
            progress.advance(task)

    log.info(f"Saved {len(slides)} slide feature files to {outdir}")

@contextmanager 
def matplotlib_backend(backend): 
    import matplotlib 
    original_backend = matplotlib.get_backend() 
    try: 
        matplotlib.use(backend) 
        yield 
    finally: 
        matplotlib.use(original_backend) 
