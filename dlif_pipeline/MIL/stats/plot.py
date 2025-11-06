
import os
import warnings

from typing import TYPE_CHECKING, List, Optional

import numpy as np
from sklearn import metrics
from MIL import errors as errors
from MIL.util import matplotlib_backend
if TYPE_CHECKING:
    import neptune.new as neptune

def scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    data_dir: str,
    name: str = '_plot',
    save_plots: bool = False,
    neptune_run: Optional["neptune.Run"] = None
) -> List[float]:
    """Generate and save scatter plots, and calculate R^2 (coefficient
    of determination) for each outcome.

    Args:
        y_true (np.ndarray): 2D array of labels. Observations are in first
            dimension, second dim is the outcome.
        y_pred (np.ndarray): 2D array of predictions.
        data_dir (str): Path to directory in which to save plots.
        name (str, optional): Label for filename. Defaults to '_plot'.
        save_plots (bool, optional): Save scatter plots as PNG files. Defaults to False.
        neptune_run (optional): Neptune Run. If provided, will upload plot.

    Returns:
        List[float]:    R squared for each outcome.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    if y_true.shape != y_pred.shape:
        m = f"Shape mismatch: y_true {y_true.shape} y_pred: {y_pred.shape}"
        raise errors.StatsError(m)
    if y_true.shape[0] < 2:
        raise errors.StatsError("Only one observation provided, need >1")
    r_squared = []

    # Subsample to n=1000 for plotting
    if y_true.shape[0] > 1000:
        idx = np.random.choice(range(y_true.shape[0]), 1000)
        yt_sub = y_true[idx]
        yp_sub = y_pred[idx]
    else:
        yt_sub = y_true
        yp_sub = y_pred

    # Perform scatter for each outcome
    with matplotlib_backend('Agg'):
        for i in range(y_true.shape[1]):
            r_squared += [metrics.r2_score(y_true[:, i], y_pred[:, i])]
            if save_plots:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    p = sns.jointplot(x=yt_sub[:, i], y=yp_sub[:, i], kind="reg")
                p.set_axis_labels('y_true', 'y_pred')
                plt.savefig(os.path.join(data_dir, f'Scatter{name}-{i}.png'))
            if save_plots and neptune_run:
                neptune_run[f'results/graphs/Scatter{name}-{i}'].upload(
                    os.path.join(data_dir, f'Scatter{name}-{i}.png')
                )
            plt.close()
    return r_squared

def roc(
    fpr: np.ndarray,
    tpr: np.ndarray,
    label: Optional[str] = None
):
    import matplotlib.pyplot as plt
    plt.clf()
    plt.title('ROC Curve')
    plt.plot(fpr, tpr, 'b', label=label)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('TPR')
    plt.xlabel('FPR')

def prc(
    precision: np.ndarray,
    recall: np.ndarray,
    label: Optional[str] = None
):
    import matplotlib.pyplot as plt
    plt.clf()
    plt.title('Precision-Recall Curve')
    plt.plot(precision, recall, 'b', label=label)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Recall')
    plt.xlabel('Precision')