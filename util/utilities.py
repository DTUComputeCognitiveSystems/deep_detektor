from pathlib import Path

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
        fig.savefig(str(path) + '.pdf', format='pdf', pad_inches=0, **options)  # , dpi=axes_dpi
