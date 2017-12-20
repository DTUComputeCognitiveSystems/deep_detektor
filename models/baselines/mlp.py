import numpy as np
import tensorflow as tf

from util.utilities import get_next_bacth
from models.model_base import DetektorModel
from math import ceil


class MLP(DetektorModel):
    @classmethod
    def _class_name(cls):
        return "MLP"

    def __init__(self, tensor_provider, hidden_units=2, learning_rate=0.001, display_step=1,
                 training_epochs=20, verbose=False, use_bow=True, use_embedsum=False, results_path=None,
                 class_weights=np.array([1.0, 1.0]), batch_size=None, batch_strategy="full",
                 name_formatter="{}"):
        """
        :param TensorProvider tensor_provider:
        :param int hidden_units:
        :param float learning_rate:
        :param int training_epochs:
        :param bool verbose:
        """

        # Settings
        self.display_step = display_step
        self.learning_rate = learning_rate
        self.hidden_units = hidden_units
        self.training_epochs = training_epochs
        self.verbose = verbose
        self.class_weights = np.array(class_weights)
        self.use_bow = use_bow
        self.use_embedsum = use_embedsum
        self.batch_size = batch_size
        self.batch_strategy = batch_strategy

        # Initialize super (and make automatic settings-summary)
        super().__init__(results_path, save_type="tf", name_formatter=name_formatter)

        self.num_features = self.x = self.y = self.Wxz = self.bz = self.Wzy = self.by = self.z = self.pred = \
            self.cost = self.optimizer = None

    def initialize_model(self, tensor_provider):
        # Get number of features
        self.num_features = tensor_provider.input_dimensions(bow=self.use_bow,
                                                             embedding_sum=self.use_embedsum)

        ####
        # Build model

        # Use model's graph
        with self._tf_graph.as_default():
            # tf Graph Input
            self.x = tf.placeholder(tf.float32, [None, self.num_features])
            self.y = tf.placeholder(tf.float32, [None, ])  # binary classification

            # Set model weights
            self.Wxz = tf.Variable(tf.zeros([self.num_features, self.hidden_units]))
            self.bz = tf.Variable(tf.zeros([self.hidden_units]))
            self.Wzy = tf.Variable(tf.zeros([self.hidden_units, 1]))
            self.by = tf.Variable(tf.zeros([1]))

            # Construct model
            self.z = tf.nn.relu(tf.matmul(self.x, self.Wxz) + self.bz)
            self.pred = tf.nn.sigmoid(tf.matmul(self.z, self.Wzy) + self.by)  # sigmoid

            # Minimize error using cross entropy (with class weights)
            self.cost = tf.reduce_mean(-self.class_weights[1] * self.y * tf.log(self.pred)
                                       - self.class_weights[0] * (1 - self.y) * tf.log(1 - self.pred))

            # Gradient Descent
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)

            # Run the initializer
            self._sess.run(tf.global_variables_initializer())

    def _fit(self, tensor_provider, train_idx, y, verbose=0):
        if verbose:
            print(verbose * " " + "Fitting {}".format(self.name))
            verbose += 2

        # Get data
        x = tensor_provider.load_concat_input_tensors(data_keys_or_idx=train_idx,
                                                      bow=self.use_bow, embedding_sum=self.use_embedsum)
        if not isinstance(x, np.ndarray):
            x = x.todense()

        # Training cycle
        for epoch in range(self.training_epochs):
            if self.batch_strategy == "full" or self.batch_size is None:
                _, c = self._sess.run([self.optimizer, self.cost], feed_dict={self.x: x,
                                                                              self.y: y})
            else:
                n_updates = int(ceil(x.shape[0] / self.batch_size))
                for n in range(n_updates):
                    x_batch, y_batch = get_next_bacth(data=x, labels=y,
                                                      batch_size=self.batch_size,
                                                      strategy=self.batch_strategy)
                    _, c = self._sess.run([self.optimizer, self.cost], feed_dict={self.x: x_batch,
                                                                                  self.y: y_batch})

                # Calculate cost on entire training data
                c = self._sess.run([self.cost], feed_dict={self.x: x, self.y: y})
                c = c[0]

            # Display logs per epoch step
            if verbose:
                if (epoch + 1) % self.display_step == 0 and verbose:
                    print(verbose * " " + "Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))

        if verbose:
            print(verbose * " " + "Optimization Finished!")

    def predict(self, tensor_provider, predict_idx, additional_fetch=None):
        # Get data
        input_tensor = tensor_provider.load_concat_input_tensors(data_keys_or_idx=predict_idx,
                                                                 bow=self.use_bow, embedding_sum=self.use_embedsum)

        # Feeds
        feed_dict = {
            self.x: input_tensor,
        }

        # Do prediction
        if additional_fetch is None:
            predictions = self._sess.run(self.pred, feed_dict=feed_dict)
        else:
            predictions = self._sess.run(self.pred + additional_fetch, feed_dict=feed_dict)

        # Binary conversion
        binary_predictions = predictions > 0.5

        return predictions, binary_predictions

    def summary_to_string(self):
        result_str = ""
        result_str += self.name + "\n"
        result_str += "Num input features: %s\n" % self.num_features
        result_str += "Num hidden units: %i\n" % self.hidden_units
        result_str += "Class weights in cost-fun: (%f,%f)\n" % (self.class_weights[0], self.class_weights[1])
        result_str += "Learning rate: %f  \n" % self.learning_rate
        result_str += "Num training epochs: %i  \n" % self.training_epochs
        result_str += "Using BoW: %i  \n" % self.use_bow
        result_str += "Using Embedsum: %i  \n" % self.use_embedsum
        result_str += "Batch sttrategy: {} \n".format(self.batch_strategy)
        if self.batch_size is not None:
            result_str += "Batch Size: %i \n" % self.batch_size
        return result_str
