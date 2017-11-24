from util.tensor_provider import TensorProvider
import tensorflow as tf

# TODO: Make methods that creates the feed_dictionary for a model, given a tensor_provider and indices


class DetektorModel:
    def __init__(self):
        # Make graph and session
        self._tf_graph = tf.Graph()
        self._sess = tf.Session(graph=self._tf_graph)

    @classmethod
    def name(cls):
        raise NotImplementedError

    def fit(self, tensor_provider, train_idx, verbose=0, **kwargs):
        """
        :param TensorProvider tensor_provider:
        :param list train_idx:
        :param int verbose:
        :param kwargs: Model-specific training-settings.
        :return:
        """
        raise NotImplementedError

    def predict(self, tensor_provider, predict_idx, additional_fetch=None):
        raise NotImplementedError
