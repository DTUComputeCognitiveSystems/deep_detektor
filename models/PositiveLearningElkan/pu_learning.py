import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression as LogRegSK
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, KFold
from util.tensor_provider import TensorProvider
from util.utilities import get_next_bacth

from models.model_base import DetektorModel
from math import ceil

import matplotlib.pyplot as plt


class PULogisticRegressionSK(DetektorModel):
    @classmethod
    def _class_name(cls):
        return "PU_LogisticRegression_SK"

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
        # self.model = LogRegSK(verbose=self.verbose)
        self.constant_c = 0.5
        self.model = SGDClassifier(loss='log', tol=1e-5)

    def _fit(self, tensor_provider, train_idx, y, verbose=0, test_size=0.2):
        if verbose:
            print(verbose * " " + "Fitting {}".format(self.name))
            verbose += 2

        if test_size <= 0 or test_size >= 1:
            print('Error, test size has to be a value between 0 and 1')
            return

        # Get training data
        x = tensor_provider.load_concat_input_tensors(data_keys_or_idx=train_idx,
                                                      bow=self.use_bow,
                                                      embedding_sum=self.use_embedsum)

        # Fetch data
        if not isinstance(x, np.ndarray):
            x = x.todense()

        # Training cycle

        # 1. Train model to learn c = P(s=1|y=1)
        positive_idx = np.where(y == True)[0]
        N_holdout = int(np.round(len(positive_idx) * test_size))
        perm_idx = np.random.permutation(positive_idx)

        positive_train_idx = np.ones([len(y), ]).astype(bool)
        positive_train_idx[perm_idx[N_holdout:]] = False
        positive_val_idx = np.zeros([len(y), ]).astype(bool)
        positive_val_idx[perm_idx[:N_holdout]] = True

        train_idx = (y == False) + positive_train_idx

        print('%i == %i' % (N_holdout, sum(positive_val_idx)))

        self.model.fit(x[train_idx], y[train_idx])
        hold_out_predictions = self.model.predict_proba(x[positive_val_idx])
        c = np.mean(hold_out_predictions[:, 1])
        self.constant_c = c

        print(c)
        plt.figure()
        plt.hist(hold_out_predictions[:, 1], bins=20)
        # 2. Scale probabilities, so 0.5 it the optimal decision boundary?
        # 3. Duplicate negative examples, give all examples a probability weight
        new_x = np.concatenate((x, x[y == False, :]), axis=0)
        w_unlabeled = self.model.predict_proba(x[y == False, :])[:, 1]
        w_unlabeled = (1.0 - self.constant_c) / self.constant_c * (w_unlabeled / (1.0 - w_unlabeled))

        y_old = y.copy()
        y_new = np.ones(x[y == False, :].shape[0], ) * True
        weights_old = np.ones(x.shape[0], )
        weights_old[y == False] = 1.0 - w_unlabeled
        weights_new = w_unlabeled
        # 4. Train new classifier
        y_pu = np.concatenate((y_old, y_new), axis=0)
        w_pu = np.concatenate((weights_old, weights_new), axis=0)

        self.model.fit(new_x, y_pu, sample_weight=w_pu)
        # 5. If necessary scale probabilities

        plt.figure()
        plt.hist(hold_out_predictions[:, 1], bins=20)

        if verbose:
            print(verbose * " " + "Optimization Finished!")

    def predict(self, tensor_provider, predict_idx, additional_fetch=None):
        # Get data
        input_tensor = tensor_provider.load_concat_input_tensors(data_keys_or_idx=predict_idx,
                                                                 bow=self.use_bow, embedding_sum=self.use_embedsum)

        # Do prediction
        predictions = self.model.predict_proba(input_tensor)
        predictions = predictions[:, 1]  # / self.constant_c
        # predictions = (1-self.constant_c)/self.constant_c * \
        #              (predictions/(1-predictions))

        # Binary conversion
        binary_predictions = predictions > 0.5
        return predictions, binary_predictions

    def summary_to_string(self):
        result_str = ""
        result_str += self.name + "\n"
        result_str += "Num input features: %i\n" % self.num_features
        result_str += "Using BoW: %i  \n" % self.use_bow
        result_str += "Using Embedsum: %i  \n" % self.use_embedsum
        return result_str
