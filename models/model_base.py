from pathlib import Path

from util.tensor_provider import TensorProvider
import tensorflow as tf
from util.utilities import ensure_folder


# TODO: Make methods that creates the feed_dictionary for a model, given a tensor_provider and indices


class DetektorModel:
    def __init__(self, results_path):
        # Make graph and session
        self._tf_graph = tf.Graph()
        self._sess = tf.Session(graph=self._tf_graph)

        # Set path
        if results_path is not None:
            self.results_path = Path(results_path, self.name())
            ensure_folder(self.results_path)
        else:
            self.results_path = None

    @classmethod
    def name(cls):
        raise NotImplementedError

    def initialize_model(self, tensor_provider):
        raise NotImplementedError

    def fit(self, tensor_provider, train_idx, verbose=0):
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
        raise NotImplementedError