

class DetektorModel:
    def __init__(self, name):
        self.name = name

    def fit(self, data, sess, indentation=0, is_batch=False):
        raise NotImplementedError

    def predict(self, data, sess):
        raise NotImplementedError
