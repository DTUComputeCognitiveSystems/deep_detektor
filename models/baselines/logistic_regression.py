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
        # Get number of features
        self.num_features = tensor_provider.input_dimensions(word_embedding=False,
                                                             pos_tags=False,
                                                             char_embedding=False,
                                                             bow=True)

        # Settings
        self.learning_rate = learning_rate
        self.training_epochs = training_epochs
        self.verbose = verbose

        ####
        # Build model

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

        # Initialize the variables (i.e. assign their default value)
        self.initializer = tf.global_variables_initializer()

    def fit(self, tensor_provider, train_idx, sess, verbose=0, display_step=1, **kwargs):
        if verbose:
            print(verbose * " " + "Fitting {}".format(self.name()))
            verbose += 2

        # Get training data
        data = tensor_provider.load_data_tensors(train_idx,
                                                 word_embedding=False,
                                                 char_embedding=False,
                                                 pos_tags=False)

        # Run the initializer
        sess.run(self.initializer)

        # Fetch data
        x = data['bow']
        if not isinstance(x, np.ndarray):
            x = x.todense()
        y = data['labels']

        # Training cycle
        for epoch in range(self.training_epochs):
            _, c = sess.run([self.optimizer, self.cost], feed_dict={self.x: x,
                                                                    self.y: y})
            # Display logs per epoch step
            if verbose:
                if (epoch + 1) % display_step == 0 and self.verbose:
                    print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))

        if verbose:
            print(verbose * " " + "Optimization Finished!")

    def predict(self, data, sess):
        x = data['bow']
        if not isinstance(x, np.ndarray):
            x = x.todense()
        ll = sess.run(self.pred, feed_dict={self.x: x})
        return ll > 0.5
