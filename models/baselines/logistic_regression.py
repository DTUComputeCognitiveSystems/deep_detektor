import numpy as np
import tensorflow as tf

from util.tensor_provider import TensorProvider

from models.model_base import DetektorModel


class LogisticRegression(DetektorModel):
    @classmethod
    def name(cls):
        return "LogisticRegression"

    def __init__(self, tensor_provider, learning_rate=0.001, training_epochs=10, verbose=False):
        """
        :param TensorProvider tensor_provider:
        :param float learning_rate:
        :param int training_epochs:
        :param bool verbose:
        """
        super().__init__()

        # Get number of features
        self.num_features = tensor_provider.input_dimensions(bow=True)

        # Settings
        self.learning_rate = learning_rate
        self.training_epochs = training_epochs
        self.verbose = verbose

        ####
        # Build model

        # Use model's graph
        with self._tf_graph.as_default():
            # tf Graph Input
            self.x = tf.placeholder(tf.float32, [None, self.num_features])
            self.y = tf.placeholder(tf.float32, [None, ])  # binary classification

            # Set model weights
            self.W = tf.Variable(tf.zeros([self.num_features, 1]))
            self.b = tf.Variable(tf.zeros([1]))

            # Construct model
            self.pred = tf.nn.sigmoid(tf.matmul(self.x, self.W) + self.b)  # sigmoid

            # Minimize error using cross entropy
            self.cost = tf.reduce_mean(-self.y * tf.log(self.pred) - (1 - self.y) * tf.log(1 - self.pred))

            # Gradient Descent
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)

            # Run the initializer
            self._sess.run(tf.global_variables_initializer())

    def fit(self, tensor_provider, train_idx, verbose=0, display_step=1, **kwargs):
        if verbose:
            print(verbose * " " + "Fitting {}".format(self.name()))
            verbose += 2

        # Get training data
        data = tensor_provider.load_data_tensors(train_idx, bow=True, labels=True)

        # Fetch data
        x = data['bow']
        if not isinstance(x, np.ndarray):
            x = x.todense()
        y = data['labels']

        # Training cycle
        for epoch in range(self.training_epochs):
            _, c = self._sess.run([self.optimizer, self.cost], feed_dict={self.x: x,
                                                                          self.y: y})
            # Display logs per epoch step
            if verbose:
                if (epoch + 1) % display_step == 0 and verbose:
                    print(verbose * " " + "Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))

        if verbose:
            print(verbose * " " + "Optimization Finished!")

    def predict(self, tensor_provider, predict_idx, additional_fetch=None, binary=True):
        # Get data
        input_tensor = tensor_provider.load_concat_input_tensors(data_keys_or_idx=predict_idx,
                                                                 bow=True)

        # Feeds
        feed_dict = {
            self.x: input_tensor,
        }

        # Do prediction
        if additional_fetch is None:
            predictions = self._sess.run(self.pred, feed_dict=feed_dict)
        else:
            predictions = self._sess.run(self.pred + additional_fetch, feed_dict=feed_dict)

        # Optional binary conversion
        if binary:
            predictions = predictions > 0.5

        return predictions
