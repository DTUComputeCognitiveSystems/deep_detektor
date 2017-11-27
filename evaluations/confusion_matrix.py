from evaluations.evaluation_base import Evaluation
import numpy as np


class TruePositives(Evaluation):
    def __call__(self, y_true, y_pred, y_pred_binary):
        return sum(np.array(y_true) * np.array(y_pred))

    def name(self):
        return "TP"


class TrueNegatives(Evaluation):
    def __call__(self, y_true, y_pred, y_pred_binary):
        return sum((1 - np.array(y_true)) * (1 - np.array(y_pred_binary)))

    def name(self):
        return "TN"


class FalsePositives(Evaluation):
    def __call__(self, y_true, y_pred, y_pred_binary):
        return sum((1 - np.array(y_true)) * np.array(y_pred_binary))

    def name(self):
        return "FP"


class FalseNegatives(Evaluation):
    def __call__(self, y_true, y_pred, y_pred_binary):
        return sum(np.array(y_true) * (1 - np.array(y_pred_binary)))

    def name(self):
        return "FN"


class PredictedPositives(Evaluation):
    def __call__(self, y_true, y_pred, y_pred_binary):
        return sum(np.array(y_pred_binary))

    def name(self):
        return "PredP"


class PredictedNegatives(Evaluation):
    def __call__(self, y_true, y_pred, y_pred_binary):
        return sum(1 - np.array(y_pred_binary))

    def name(self):
        return "PredN"


class DataPositives(Evaluation):
    def __call__(self, y_true, y_pred, y_pred_binary):
        return sum(np.array(y_true))

    def name(self):
        return "DataP"


class DataNegatives(Evaluation):
    def __call__(self, y_true, y_pred, y_pred_binary):
        return sum(1 - np.array(y_true))

    def name(self):
        return "DataN"


class Samples(Evaluation):
    def __call__(self, y_true, y_pred, y_pred_binary):
        return len(y_true)

    def name(self):
        return "Samples"
