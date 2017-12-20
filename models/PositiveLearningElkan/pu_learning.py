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
        self.model = SGDClassifier(loss='log', tol=1e-5)


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

        #K=10
        #cv = KFold(n_splits=K)
        #c = []
        #for train_idx, val_idx in cv.split(x, y):
        #    self.model.fit(x[train_idx],y[train_idx])
        #    hold_out_predictions = self.model.predict_proba(x[val_idx])
        #
        #    c.append(np.mean(hold_out_predictions[:,1] * (y[val_idx] == True)))

        #print('Est. p(s=1 | x) for %i CV folds:'%(K))
        #print(c)
        #print('''mean(c) \t= {:1.4f} \nmedian(c) \t= {:1.4f} \nsd(c) \t= {:1.4f}'''.\
        #      format(np.mean(c),np.median(c),np.sqrt(np.var(c))))

        #self.constant_c = np.median(c)
        hold_out_ratio = 0.1
        positives = np.where(y == True)[0]
        hold_out_size = int(np.ceil(len(positives) * hold_out_ratio))

        np.random.shuffle(positives)
        hold_out = positives[:hold_out_size]
        X_hold_out = x[hold_out, :]
        X = np.delete(x, hold_out, 0)
        y_c = np.delete(y, hold_out)

        self.model.fit(X, y_c)

        hold_out_predictions = self.model.predict_proba(X_hold_out)

        c = np.mean(hold_out_predictions[:, 1])
        print(c)
        self.constant_c = c


        new_x = np.concatenate((x, x[y == False, :]), axis=0)
        w_unlabeled = self.model.predict_proba(x[y == False, :])[:,1]
        w_unlabeled = (1.0-self.constant_c)/self.constant_c * (w_unlabeled / (1.0-w_unlabeled))

        y_old = y.copy()
        y_new = np.ones(x[y == False, :].shape[0],)*True
        weights_old = np.ones(x.shape[0],)
        weights_old[y == False] = 1.0-w_unlabeled
        weights_new  = w_unlabeled

        y_pu = np.concatenate((y_old,y_new), axis=0)
        w_pu = np.concatenate((weights_old,weights_new), axis=0)

        self.model.fit(new_x, y_pu, sample_weight=w_pu)

        if verbose:
            print(verbose * " " + "Optimization Finished!")

    def predict(self, tensor_provider, predict_idx, additional_fetch=None):
        # Get data
        input_tensor = tensor_provider.load_concat_input_tensors(data_keys_or_idx=predict_idx,
                                                                 bow=self.use_bow, embedding_sum=self.use_embedsum)

        # Do prediction
        predictions = self.model.predict_proba(input_tensor)
        predictions = predictions[:, 1] #/ self.constant_c
        #predictions = (1-self.constant_c)/self.constant_c * \
        #              (predictions/(1-predictions))

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