import warnings
from pathlib import Path
import pickle
from typing import Sized, Iterable

from util.tensor_provider import TensorProvider
import tensorflow as tf
from util.utilities import ensure_folder


# TODO: Make methods that creates the feed_dictionary for a model, given a tensor_provider and indices


class DetektorModel:
    def __init__(self, results_path, save_type=None, summary_ignore=set(), name_formatter="{}"):
        # Make graph and session
        self._tf_graph = tf.Graph()
        self._sess = tf.Session(graph=self._tf_graph)
        self.save_type = save_type
        self.model = None
        self._auto_summary_keys = None
        name_formatter = name_formatter if name_formatter is not None else "{}"
        self._name = name_formatter.format(self._class_name())

        # Create automatic summary dictionary
        self._create_autosummary_dict(summary_ignore)

        # Set path
        if results_path is not None:
            self.results_path = self.create_model_path(results_path=results_path)
            ensure_folder(self.results_path)
        else:
            self.results_path = None

    def create_model_path(self, results_path, use_settings=False, auto_enumerate=True):
        path = Path(results_path, self.name)
        if use_settings:
            path = Path(results_path, self.generate_settings_name())

        if auto_enumerate:
            new_path = Path(path.parent, path.name + "_01")
            nr = 1
            while new_path.exists():
                nr += 1
                new_path = Path(path.parent, path.name + "_{:02d}".format(nr))

            path = new_path

        return path

    def _attribute_name_list(self):
        return []

    def generate_settings_name(self):
        attribute_name_list = self._attribute_name_list()

        settings_list = []
        for attribute, name, *special_handler in attribute_name_list:
            value = getattr(self, attribute)

            if isinstance(value, bool):
                settings_list.append("{}".format(name) if value else "")
            elif isinstance(value, (int, float)):
                settings_list.append("{}-{}".format(name, value).replace(".", "'"))
            elif isinstance(value, Sized) and isinstance(value, Iterable):
                if len(value) == 0:
                    settings_list.append("{}-{}".format(name, "|"))
                else:
                    settings_list.append("{}-{}".format(name,
                                                        "_".join([str(val).replace(".", "'") for val in value])))
            elif isinstance(value, type):
                settings_list.append("{}-{}".format(name, value.__name__))
            else:
                raise ValueError("Can't handle type '{}' of field '{}'.".format(type(value).__name__, name))

        settings_str = "_".join([self.name] + settings_list)

        return settings_str

    @property
    def name(self):
        return self._name

    def set_name(self, new_name):
        self._name = new_name

    @classmethod
    def _class_name(cls):
        raise NotImplementedError

    def initialize_model(self, tensor_provider):
        raise NotImplementedError

    def fit(self, tensor_provider, train_idx, verbose=0, y=None):
        """
        :param TensorProvider tensor_provider:
        :param list train_idx:
        :param int verbose:
        :param list | np.ndarray y:
        :return:
        """
        # Load labels (just ensuring that this works for all methods)
        if y is None:
            y = tensor_provider.load_labels(data_keys_or_idx=train_idx)

        # Run specific model's run-method
        self._fit(tensor_provider=tensor_provider,
                  train_idx=train_idx,
                  y=y,
                  verbose=verbose)

    def _fit(self, tensor_provider, train_idx, y, verbose=0):
        """
        :param TensorProvider tensor_provider:
        :param list train_idx:
        :param int verbose:
        :return:
        """
        raise NotImplementedError

    def predict(self, tensor_provider, predict_idx, additional_fetch=None):
        raise NotImplementedError

    def summary_to_string(self):
        # Coverts all relevant model properties to be printed
        warnings.warn("model.summary_to_string() not implemented.", UserWarning)

        return ""

    def _create_autosummary_dict(self, summary_ignore):

        # Ignore fields
        not_allowed = {"self", "tensor_provider", "not_allowed", "save_type", "model"}
        not_allowed.update(summary_ignore)

        # Fetch keys for summary
        autosummary = [key for key in self.__dict__.keys()
                       if not key.startswith("_") and key not in not_allowed]

        # Remember
        self._auto_summary_keys = autosummary

    def autosummary_str(self):
        summary_str = self.name
        for key in sorted(self._auto_summary_keys):
            summary_str += "\n    {} : {}".format(key, getattr(self, key))
        return summary_str

    def save_model(self):
        if self.results_path is not None:
            if self.save_type == "tf":
                print("Saving model. ")

                # Use model's graph
                with self._tf_graph.as_default():
                    # Complete path
                    checkpoint_path = Path(self.results_path, "Checkpoint", 'model.checkpoint')

                    # Create folder if needed
                    ensure_folder(checkpoint_path)

                    # Save session to path
                    tf.train.Saver(tf.trainable_variables()).save(self._sess, str(checkpoint_path))

            if self.save_type == "sk":
                print("Saving model. ")

                pickle.dump(self.model, Path(self.results_path, "model.p").open("wb"))

    def load_model(self, results_path):
        if self.save_type == "tf":
            # Use model's graph
            with self._tf_graph.as_default():
                # Complete path
                checkpoint_path = Path(results_path, "Checkpoint", 'model.checkpoint')

                # Load session from path
                tf.train.Saver(tf.trainable_variables()).restore(self._sess, str(checkpoint_path))

        if self.save_type == "sk":
            self.model = pickle.load(Path(self.results_path, "model.p").open("rb"))
