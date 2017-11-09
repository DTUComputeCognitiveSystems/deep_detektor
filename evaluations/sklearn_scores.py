from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import warnings
from evaluations.evaluation_base import Evaluation


class F1(Evaluation):
    def __call__(self, y_true, y_pred):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return f1_score(y_true=y_true, y_pred=y_pred)

    def name(self):
        return "F1"


class Accuracy(Evaluation):
    def __call__(self, y_true, y_pred):
        return accuracy_score(y_true=y_true, y_pred=y_pred)

    def name(self):
        return "Accuracy"


class Precision(Evaluation):
    def __call__(self, y_true, y_pred):
        return precision_score(y_true=y_true, y_pred=y_pred)

    def name(self):
        return "Precision"


class Recall(Evaluation):
    def __call__(self, y_true, y_pred):
        return recall_score(y_true=y_true, y_pred=y_pred)

    def name(self):
        return "Recall"
