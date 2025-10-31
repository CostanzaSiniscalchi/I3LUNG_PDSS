"""Module minimal pour le MIL, extrait de la classe Project de MIL."""

import os
from typing import Optional, Union, List
import csv
from MIL.dataset import Dataset
from typing import ( Any, List, Optional,
                    Union)
from MIL.util import log, exists, is_project, load_json, relative_path, path_to_name
from os.path import join
from MIL import errors
import MIL.mil as mil

# 
class Project: 
    """Assists with project organization and execution of common tasks.""" 
 
    def __init__( 
        self, root: str, 
        use_neptune: bool = False, 
        create: bool = False, 
        **kwargs 
    ) -> None: 
        """Load or create a project at a given directory. 
 
        If a project does not exist at the given root directory, one can be 
        created if a project configuration was provided via keyword arguments. 
 
        *Create a project:* 
 
        .. code-block:: python 
 
            from MIL.project import Project
            P = sf.Project('/project/path', name=..., ...) 
 
        *Load an existing project:* 
 
        .. code-block:: python 
 
            P = sf.Project('/project/path') 
 
        Args: 
            root (str): Path to project directory. 
 
        Keyword Args: 
            name (str): Project name. Defaults to 'MyProject'. 
            annotations (str): Path to annotations CSV file. 
                Defaults to './annotations.csv' 
            dataset_config (str): Path to dataset configuration JSON file. 
                Defaults to './datasets.json'. 
            sources (list(str)): List of dataset sources to include in project. 
                Defaults to 'source1'. 
            models_dir (str): Path to directory in which to save models. 
                Defaults to './models'. 
            eval_dir (str): Path to directory in which to save evaluations. 
                Defaults to './eval'. 
 
        Raises: 
                MIL.errors.ProjectError: if project folder does not exist, 
                or the folder exists but kwargs are provided. 
 
        """ 
        self.root = root 
        if is_project(root) and kwargs: 
                raise errors.ProjectError(f"Project already exists at {root}") 
        elif is_project(root): 
            self._load(root) 
        else:
            self._settings = {
            'name': 'MyProject',
            'annotations': './annotations.csv', 
            'dataset_config': './datasets.json',
            # 'models_dir': './models',  # Supprimez cette ligne
            'eval_dir': './eval',
            'sources': ['source1']
        }
 
        # Create directories, if not already made 
        # if not exists(self.models_dir): 
        #     os.makedirs(self.models_dir) 
        if not exists(self.eval_dir): 
            os.makedirs(self.eval_dir) 
 
        # Create blank annotations file if one does not exist 
        if not exists(self.annotations) and exists(self.dataset_config): 
            self.create_blank_annotations() 
 
        # Neptune 
        self.use_neptune = use_neptune 
        
    def create_blank_annotations(
        self,
        filename: Optional[str] = None
    ) -> None:
            """Create an empty annotations file.

            Args:
                filename (str): Annotations file destination. If not provided,
                    will use project default.

            """
            if filename is None:
                filename = self.annotations
            if exists(filename):
                raise errors.AnnotationsError(
                    f"Error creating annotations {filename}; file already exists"
                )
            if not exists(self.dataset_config):
                raise errors.AnnotationsError(
                    f"Dataset config {self.dataset_config} missing."
                )
            dataset = Dataset(
                config=self.dataset_config,
                sources=self.sources,
                tile_px=None,
                tile_um=None,
                annotations=None
            )
            all_paths = dataset.slide_paths(apply_filters=False)
            slides = [path_to_name(s) for s in all_paths]
            with open(filename, 'w') as csv_outfile:
                csv_writer = csv.writer(csv_outfile, delimiter=',')
                header = ['patient', 'dataset', 'category']
                csv_writer.writerow(header)
                for slide in slides:
                    csv_writer.writerow([slide, '', ''])
            log.info(f"Wrote annotations file to [green]{filename}")

    @property 
    def annotations(self) -> str: 
        """Path to annotations file.""" 
        return self._read_relative_path(self._settings['annotations']) 
 
    @annotations.setter 
    def annotations(self, val: str) -> None: 
        if not isinstance(val, str): 
            raise errors.ProjectError("'annotations' must be a path.") 
        self._settings['annotations'] = val 
 
    @property 
    def dataset_config(self) -> str: 
        """Path to dataset configuration JSON file.""" 
        return self._read_relative_path(self._settings['dataset_config']) 
 
    @dataset_config.setter 
    def dataset_config(self, val: str) -> None: 
        if not isinstance(val, str): 
            raise errors.ProjectError("'dataset_config' must be path to JSON.") 
        self._settings['dataset_config'] = val 
 
    @property 
    def eval_dir(self) -> str: 
        """Path to evaluation directory.""" 
        if 'eval_dir' not in self._settings: 
            log.debug("Missing eval_dir in project settings, Assuming ./eval") 
            return self._read_relative_path('./eval') 
        else: 
            return self._read_relative_path(self._settings['eval_dir']) 

    @property 
    def models_dir(self) -> str: 
        """Path to models directory.""" 
        return self._read_relative_path(self._settings['models_dir']) 

    @property 
    def sources(self) -> List[str]: 
        """List of dataset sources active in this project.""" 
        if 'sources' in self._settings: 
            return self._settings['sources'] 
        elif 'datasets' in self._settings: 
            log.debug("'sources' misnamed 'datasets' in project settings.") 
            return self._settings['datasets'] 
        else: 
            raise ValueError('Unable to find project dataset sources') 

    def _load(self, path: str) -> None: 
        """Load a saved and pre-configured project from the specified path.""" 
        if is_project(path): 
            self._settings = load_json(join(path, 'settings.json')) 
        else: 
            raise errors.ProjectError('Unable to find settings.json.') 
        
    ''' def _read_relative_path(self, path: str) -> str: 
        """Convert relative path within project directory to global path.""" 
        return relative_path(path, self.root) '''
    
    def _read_relative_path(self, path: str) -> str:
        """Convert relative path within project directory to global path."""
        if os.path.isabs(path):
            return path
        return os.path.abspath(os.path.join(self.root, path))

    def dataset( 
        self, 
        tile_px: Optional[int] = None, 
        tile_um: Optional[Union[int, str]] = None, 
        *, 
        verification: Optional[str] = 'both', 
        **kwargs: Any 
    ) -> Dataset: 
        """Return a :class:`MIL.Dataset` object using project settings. 
 
        Args: 
            tile_px (int): Tile size in pixels 
            tile_um (int or str): Tile size in microns (int) or magnification 
                (str, e.g. "20x"). 
 
        Keyword Args: 
            filters (dict, optional): Dataset filters to use for 
                selecting slides. See :meth:`MIL.Dataset.filter` for 
                more information. Defaults to None. 
            filter_blank (list(str) or str, optional): Skip slides that have 
                blank values in these patient annotation columns. 
                Defaults to None. 
            min_tiles (int, optional): Min tiles a slide must have. 
                Defaults to 0. 
            config (str, optional): Path to dataset configuration JSON file. 
                Defaults to project default. 
            sources (str, list(str), optional): Dataset sources to use from 
                configuration. Defaults to project default. 
            verification (str, optional): 'tfrecords', 'slides', or 'both'. 
                If 'slides', verify all annotations are mapped to slides. 
                If 'tfrecords', check that TFRecords exist and update manifest. 
                Defaults to 'both'. 
 
        """ 
        if 'config' not in kwargs: 
            kwargs['config'] = self.dataset_config 
        if 'sources' not in kwargs: 
            kwargs['sources'] = self.sources 
        try: 
            if self.annotations and exists(self.annotations): 
                annotations = self.annotations 
            else: 
                annotations = None 
            dataset = Dataset( 
                tile_px=tile_px, 
                tile_um=tile_um, 
                annotations=annotations, 
                **kwargs 
            ) 
        except FileNotFoundError: 
            raise errors.DatasetError('No datasets configured.') 
        if verification in ('both', 'slides'): 
            log.debug("Verifying slide annotations...") 
            dataset.verify_annotations_slides() 
        if verification in ('both', 'tfrecords'): 
            log.debug("Verifying tfrecords...") 
            dataset.update_manifest() 
        return dataset 
    
    def train_mil( 
        self, 
        config: "mil.TrainerConfig", 
        train_dataset: Dataset, 
        val_dataset: Dataset, 
        outcomes: Union[str, List[str]], 
        bags: Union[str, List[str]], 
        *, 
        exp_label: Optional[str] = None, 
        outdir: Optional[str] = None, 
        **kwargs 
    ): 
        r"""Train a multi-instance learning model. 
 
        Args: 
            config (:class:`MIL.mil.TrainerConfig`): 
                Training configuration, as obtained by 
                :func:`MIL.mil.mil_config()`. 
            train_dataset (:class:`MIL.Dataset`): Training dataset. 
            val_dataset (:class:`MIL.Dataset`): Validation dataset. 
            outcomes (str): Outcome column (annotation header) from which to 
                derive category labels. 
            bags (str): Either a path to directory with \*.pt files, or a list 
                of paths to individual \*.pt files. Each file should contain 
                exported feature vectors, with each file containing all tile 
                features for one patient. 
 
        Keyword args: 
            exp_label (str): Experiment label, used for naming the subdirectory 
                in the ``{project root}/mil`` folder, where training history 
                and the model will be saved. 
            attention_heatmaps (bool): Calculate and save attention heatmaps 
                on the validation dataset. Defaults to False. 
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
                event, for training a survival. 
        """ 
        from .mil import train_mil 
        if outdir is None: 
            outdir = join(self.root, 'mil') 
 
        return train_mil( 
            config, 
            train_dataset, 
            val_dataset, 
            outcomes, 
            bags, 
            outdir=outdir, 
            exp_label=exp_label, 
            **kwargs 
        ) 
 
# -----------------------------------------------------------------------------  