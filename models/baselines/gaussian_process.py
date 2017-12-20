import numpy as np
import tensorflow as tf
from sklearn.gaussian_process import GaussianProcessClassifier
from util.tensor_provider import TensorProvider
from util.utilities import get_next_bacth

from models.model_base import DetektorModel
from math import ceil


# TODO: Make a wrapper for SKLEARN so future sk-models can be more easily included.


class GaussianProcess(DetektorModel):
    @classmethod
    def _class_name(cls):
        return "GaussianProcess"

    def __init__(self, tensor_provider,
                 use_bow=True, use_embedsum=False,
                 display_step=1, verbose=False, results_path=None,
                 kernel=None, n_restarts_optimizer=0, max_iter_predict=100, n_jobs=1,
                 name_formatter="{}"
                 ):

        # Settings
        self.display_step = display_step
        self.verbose = verbose
        self.use_bow = use_bow
        self.use_embedsum = use_embedsum

        self.kernel = kernel
        self.n_restarts_optimizer = n_restarts_optimizer
        self.max_iter_predict = max_iter_predict
        self.n_jobs = n_jobs

        self.num_features = None  # type: int

        # Initialize super (and make automatic settings-summary)
        super().__init__(results_path=results_path, save_type="sk", name_formatter=name_formatter)

    def initialize_model(self, tensor_provider):
        # Get number of features
        self.num_features = tensor_provider.input_dimensions(bow=self.use_bow,
                                                             embedding_sum=self.use_embedsum)

        # Make model
        self.model = GaussianProcessClassifier(
            kernel=self.kernel,
            n_restarts_optimizer=self.n_restarts_optimizer,
            max_iter_predict=self.max_iter_predict,
            copy_X_train=True,
            n_jobs=self.n_jobs
        )

    def _fit(self, tensor_provider, train_idx, y, verbose=0):
        if verbose:
            print(verbose * " " + "Fitting {}".format(self.name))
            verbose += 2

        # Get training data
        x = tensor_provider.load_concat_input_tensors(data_keys_or_idx=train_idx,
                                                      bow=self.use_bow,
                                                      embedding_sum=self.use_embedsum)

        # Fetch data
        if not isinstance(x, np.ndarray):
            x = x.todense()

        # Training cycle
        self.model.fit(x, y)

        if verbose:
            print(verbose * " " + "Optimization Finished!")

    def predict(self, tensor_provider, predict_idx, additional_fetch=None):
        # Get data
        input_tensor = tensor_provider.load_concat_input_tensors(data_keys_or_idx=predict_idx,
                                                                 bow=self.use_bow, embedding_sum=self.use_embedsum)

        # Do prediction
        predictions = self.model.predict_proba(input_tensor)
        predictions = predictions[:, 1]

        # Binary conversion
        binary_predictions = predictions > 0.5
        return predictions, binary_predictions

    def summary_to_string(self):
        return self.autosummary_str()
