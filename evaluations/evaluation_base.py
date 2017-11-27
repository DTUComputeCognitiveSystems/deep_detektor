class Evaluation:
    @property
    def is_single_value(self):
        return True

    def name(self):
        raise NotImplementedError

    def __call__(self, y_true, y_pred, y_pred_binary):
        raise NotImplementedError
