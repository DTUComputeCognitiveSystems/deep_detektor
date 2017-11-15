class Evaluation:
    def name(self):
        raise NotImplementedError

    def __call__(self, y_true, y_pred):
        raise NotImplementedError
