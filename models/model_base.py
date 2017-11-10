from util.tensor_provider import TensorProvider
import tensorflow as tf


class DetektorModel:
    @classmethod
    def name(cls):
        raise NotImplementedError

    def fit(self, tensor_provider, train_idx, sess, indentation=0, **kwargs):
        """
        :param TensorProvider tensor_provider:
        :param list train_idx:
        :param tf.Session sess:
        :param int indentation:
        :param kwargs: Model-specific training-settings.
        :return:
        """
        raise NotImplementedError

    def predict(self, data, sess):
        raise NotImplementedError
