import sys
from pathlib import Path
from time import sleep

import numpy as np
import matplotlib.pyplot as plt


def get_dir(path):
    """
    Returns the directory of a file, or simply the original path if the path is a directory (has no extension)
    :param Path path:
    :return: Path
    """
    extension = path.suffix
    if extension == '':
        return path
    else:
        return path.parent


def empty_folder(path: Path):
    for the_file in path.glob("*"):  # type: Path
        if the_file.is_file():
            the_file.unlink()


def ensure_folder(*arg):
    """
    Ensures the existence of a folder. If the folder does not exist it is created, otherwise nothing happens.
    :param str | Path arg: Any number of strings of Path-objects which can be combined to a path.
    """
    if len(arg) == 0:
        raise Exception("No input to ensure_folder")
    path = get_dir(Path(*arg))
    path.mkdir(parents=True, exist_ok=True)


def save_fig(path, only_png=False, only_pdf=False, dpi=None, bbox_inches="tight",
             facecolor=None):
    """
    Save figure as pdf and png, with the same
    :param facecolor:
    :param int dpi: Dots per inch for the png image
    :param bool only_pdf:
    :param bool only_png:
    :param Path path: Path to save on, with filename but without extension
    """
    ensure_folder(path.parent)
    options = dict(bbox_inches=bbox_inches)
    if facecolor is not None:
        options["facecolor"] = facecolor
    if not only_pdf:
        if dpi is None:
            plt.savefig(str(Path(str(path) + '.png')), **options)
        else:
            plt.savefig(str(Path(str(path) + '.png')), dpi=dpi, **options)
    if not only_png:
        fig = plt.gcf()

        # Try storing (files may be temporarily held by another process - e.g. a PDF-viewer)
        success = False
        e = None
        for _ in range(10):
            try:
                fig.savefig(str(path) + '.pdf', format='pdf', pad_inches=0, **options)  # , dpi=axes_dpi
                success = True
                break
            except PermissionError as e:
                sleep(0.25)
        if not success:
            raise e


def get_next_bacth(data, labels, batch_size=None, strategy="weighted_sampling"):
    """
      Get next batch for mini_batch training
      :param array-like data: input data that from which a batch should be sampled
      :param labels: data class labels
      :param int batch_size: number of examples in batch sampled
      :param string strategy: specifies what strategy to use when getting new batch
      :return data_batch and labels_batch
      """

    # If no batch size is specified return input
    if batch_size==None or strategy=="full":
        return data, labels

    # Get number of observations
    n_obs = data.shape[0]
    assert(n_obs == len(labels))

    # Based on strategy input do different things:
    if strategy == "weighted_sampling":
        # Get inverse-frequency of each class
        non_claim_if = 1.0 / sum(labels == 0)
        claim_if = 1.0 / sum(labels == 1)

        # Construct sampling weights for each observation
        sample_weights = np.empty((n_obs))
        sample_weights[labels == 0] = non_claim_if
        sample_weights[labels == 1] = claim_if
        sample_weights = sample_weights / sum(sample_weights)  # normalize to yield probabilities

        # Get indices for batch
        c_indices = np.random.choice(range(n_obs), batch_size, replace=False,
                                     p=sample_weights)

        data_batch = data[c_indices, ]
        labels_batch = labels[c_indices]
    else:
        raise NotImplementedError

    return data_batch, labels_batch


class _STDOutRedirector:

    def __init__(self, stream, path: Path):
        if isinstance(stream, _STDOutRedirector):
            raise ValueError("A _STDOutRedirector received another _STDOutRedirector as "
                             "stream (this should not happen)")

        self.stream = stream
        self.file = path.open("w")

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stream.flush()

    def close(self):
        self.file.close()
        sys.stdout = self.stream
        del self


def close_stdout_file():
    if isinstance(sys.stdout, _STDOutRedirector):
        sys.stdout.close()


def redirect_stdout_to_file(path):
    """
    Redirects the stdout to both the console and to a file (experimental).
    :param Path path: Path to log-file.
    """

    if not isinstance(sys.stdout, _STDOutRedirector):
        sys.stdout = _STDOutRedirector(sys.stdout, path)
