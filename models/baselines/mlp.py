import numpy as np
import tensorflow as tf

from models.model_base import DetektorModel


class MLP(DetektorModel):
    @classmethod
    def name(cls):
        return "MLP"

    def __init__(self, tensor_provider, hidden_units=2, learning_rate=0.001,
                 training_epochs=20, verbose=False, use_bow=False, use_embedsum=True,
                 class_weights=np.array([1.0, 1.0])):
        """
        :param TensorProvider tensor_provider:
        :param int hidden_units:
        :param float learning_rate:
        :param int training_epochs:
        :param bool verbose:
        """
        super().__init__()

        # Get number of features
        self.num_features = tensor_provider.input_dimensions(bow=use_bow, embedding_sum=use_embedsum)

        # Settings
        self.learning_rate = learning_rate
        self.hidden_units = hidden_units
        self.training_epochs = training_epochs
        self.verbose = verbose
        self.class_weights = np.array(class_weights)
        self.use_bow = use_bow
        self.use_embedsum = use_embedsum

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

    def fit(self, tensor_provider, train_idx, verbose=0, display_step=1, **kwargs):
        if verbose:
            print(verbose * " " + "Fitting {}".format(self.name()))
            verbose += 2

        # Get data
        x = tensor_provider.load_concat_input_tensors(data_keys_or_idx=train_idx,
                                                      bow=self.use_bow, embedding_sum=self.use_embedsum)
        if not isinstance(x, np.ndarray):
            x = x.todense()

        # Load labels
        y = tensor_provider.load_labels(data_keys_or_idx=train_idx)

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
