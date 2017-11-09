import abc


class DetektorModel:
    @classmethod
    def name(cls):
        raise NotImplementedError

    def fit(self, data, sess, indentation=0, is_batch=False):
        raise NotImplementedError

    def predict(self, data, sess):
        raise NotImplementedError
