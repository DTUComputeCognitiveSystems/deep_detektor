import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression as LogRegSK
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold, KFold
from util.tensor_provider import TensorProvider
from util.utilities import get_next_bacth

from models.model_base import DetektorModel
from math import ceil

class PULogisticRegressionSK(DetektorModel):
    @classmethod
    def name(cls):
        return "PU LogisticRegression (Scikit-learn)"

    def __init__(self, tensor_provider, use_bow=True, use_embedsum=False, display_step=1, verbose=False):
        """
        :param TensorProvider tensor_provider:
        :param float learning_rate:
        :param int training_epochs:
        :param bool verbose:
        """
        super().__init__(None)

        # Settings
        self.display_step = display_step
        self.verbose = verbose
        self.use_bow = use_bow
        self.use_embedsum = use_embedsum

        self.num_features = self.x = self.y = self.W = self.b = self.pred = self.cost = self.optimizer = None

    def initialize_model(self, tensor_provider):
        # Get number of features
        self.num_features = tensor_provider.input_dimensions(bow=self.use_bow,
                                                             embedding_sum=self.use_embedsum)
        #self.model = LogRegSK(verbose=self.verbose)
        self.constant_c = 0.5
        self.model = SGDClassifier(loss='log')


    def fit(self, tensor_provider, train_idx, verbose=0):
        if verbose:
            print(verbose * " " + "Fitting {}".format(self.name()))
            verbose += 2

        # Get training data
        x = tensor_provider.load_concat_input_tensors(data_keys_or_idx=train_idx,
                                                      bow=self.use_bow,
                                                      embedding_sum=self.use_embedsum)

        # Fetch data
        if not isinstance(x, np.ndarray):
            x = x.todense()

        # Load labels
        y = tensor_provider.load_labels(data_keys_or_idx=train_idx)

        # Training cycle
        #self.model.fit(x,y)

        K=10
        cv = KFold(n_splits=K)
        c = []
        for train_idx, val_idx in cv.split(x):
            self.model.fit(x[train_idx],y[train_idx])
            hold_out_predictions = self.model.predict_proba(x[val_idx])

            c.append(np.mean(hold_out_predictions[:,0] * (y[val_idx] == True)))

        print('Est. p(s=1 | x) for %i CV folds:'%(K))
        print(c)
        print('''mean(c) \t= {:1.4f} \nmedian(c) \t= {:1.4f} \nsd(c) \t= {:1.4f}'''.\
              format(np.mean(c),np.median(c),np.sqrt(np.var(c))))

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
        result_str = ""
        result_str += self.name() + "\n"
        result_str += "Num input features: %i\n" % self.num_features
        result_str += "Using BoW: %i  \n" % self.use_bow
        result_str += "Using Embedsum: %i  \n" % self.use_embedsum
        return result_str