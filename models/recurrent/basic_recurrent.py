from models.model_base import DetektorModel
import tensorflow as tf
import numpy as np

from util.tensor_provider import TensorProvider


class BasicRecurrent(DetektorModel):
    def __init__(self, tensor_provider, units=None, epsilon=1e-10):
        """
        :param TensorProvider tensor_provider:
        :param units:
        :param verbose:
        """
        super().__init__()

        # Use model's graph
        with self._tf_graph.as_default():

            self.hidden_units = units if units is not None else [100, 50]

            # Get number of features
            self.num_features = tensor_provider.input_dimensions(word_embedding=True,
                                                                 pos_tags=True,
                                                                 char_embedding=True)

            # Model inputs
            self.inputs = tf.placeholder(tf.float32, shape=[None, None, self.num_features], name='input')
            self.input_lengths = tf.placeholder(tf.int32, shape=[None], name='input_length')
            self.truth = tf.placeholder(tf.float32, [None, ])

            # Recurrent layer
            with tf.name_scope("recurrent_layer"):
                self._rec_cell = tf.nn.rnn_cell.GRUCell(num_units=self.hidden_units[0])
                self.rec_cell_outputs, self.rec_cell_state = tf.nn.dynamic_rnn(cell=self._rec_cell,
                                                                               inputs=self.inputs,
                                                                               sequence_length=self.input_lengths,
                                                                               dtype=tf.float32)

            # Fully connected layer1
            with tf.name_scope("ff_layer1"):
                self._ff1_m = tf.Variable(tf.truncated_normal([self.hidden_units[0], self.hidden_units[1]],
                                                              stddev=np.sqrt(self.hidden_units[1])),
                                          name="ff1_m")
                self._ff1_b = tf.Variable(tf.truncated_normal([self.hidden_units[1]],
                                                              stddev=np.sqrt(self.hidden_units[1])),
                                          name="ff1_b")
                self._ff1_prod = tf.matmul(self.rec_cell_state, self._ff1_m)
                self._ff1_a = self._ff1_prod + self._ff1_b
                self.ff1_act = tf.nn.relu(self._ff1_a, name="ff1_activation")

            # Output layer
            with tf.name_scope("output_layer"):
                self._ffout_m = tf.Variable(tf.truncated_normal([self.hidden_units[1], 1],
                                                                stddev=np.sqrt(self.hidden_units[0])),
                                            name="ffout_m")
                self._ffout_b = tf.Variable(tf.truncated_normal([1], stddev=1),
                                            name="ffout_b")
                self._ffout_prod = tf.matmul(self.ff1_act, self._ffout_m)
                self._ffout_a = self._ffout_prod + self._ffout_b
                self.prediction = tf.squeeze(tf.nn.sigmoid(self._ffout_a, name="output"), axis=1)

            # Cost-function
            self.cost = tf.nn.softmax_cross_entropy_with_logits(labels=self.truth,
                                                                logits=tf.transpose(self._ffout_a))

            # Gradient Descent
            self.learning_rate = tf.placeholder(shape=(), dtype=tf.float32, name="learning_rate")
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)

            # Run the initializer
            self._sess.run(tf.global_variables_initializer())

    def fit(self, tensor_provider, train_idx, n_batches=10000, batch_size=40,
            verbose=0, display_step=20, **kwargs):
        """
        :param TensorProvider tensor_provider:
        :param list train_idx:
        :param int n_batches:
        :param int batch_size:
        :param int verbose:
        :param int display_step:
        :param kwargs:
        :return:
        """
        if verbose:
            print(verbose * " " + "Fitting {}".format(self.name()))
            verbose += 2

        # Get training data
        input_tensor = tensor_provider.load_concat_input_tensors(data_keys_or_idx=train_idx,
                                                                 word_embedding=True,
                                                                 char_embedding=True,
                                                                 pos_tags=True)
        output_truth = tensor_provider.load_labels(data_keys_or_idx=train_idx)
        input_lengths = tensor_provider.load_data_tensors(data_keys_or_idx=train_idx, word_counts=True)["word_counts"]
        train_idx = list(range(len(train_idx)))

        # Run training batches
        for batch_nr in range(n_batches):
            c_indices = np.random.choice(train_idx, batch_size, replace=False)
            c_inputs = input_tensor[c_indices, :, :]
            c_truth = output_truth[c_indices]
            c_input_lengths = input_lengths[c_indices]

            # Feeds
            feed_dict = {
                self.inputs: c_inputs,
                self.input_lengths: c_input_lengths,
                self.truth: c_truth,
                self.learning_rate: 0.0
            }

            # Fetching
            fetch = [self.optimizer, self.cost]

            # Run batch training
            _, c = self._sess.run(fetches=fetch, feed_dict=feed_dict)

            if verbose:
                if (batch_nr + 1) % display_step == 0 and verbose:
                    print(verbose * " " + "\tBatch {: 6d}. cost = {: 8.4f}".format(batch_nr + 1, c[0]))

    @classmethod
    def name(cls):
        return "BasicRecurrent"

    def predict(self, tensor_provider, predict_idx, additional_fetch=None, binary=True):
        # Get data
        input_tensor = tensor_provider.load_concat_input_tensors(data_keys_or_idx=predict_idx,
                                                                 word_embedding=True,
                                                                 char_embedding=True,
                                                                 pos_tags=True)
        input_lengths = tensor_provider.load_data_tensors(data_keys_or_idx=predict_idx, word_counts=True)["word_counts"]

        # Feeds
        feed_dict = {
            self.inputs: input_tensor,
            self.input_lengths: input_lengths,
        }

        # Do prediction
        if additional_fetch is None:
            predictions = self._sess.run(self.prediction, feed_dict=feed_dict)
        else:
            predictions = self._sess.run([self.prediction] + additional_fetch, feed_dict=feed_dict)

        # Optional binary conversion
        if binary:
            predictions = predictions > 0.5

        return predictions
