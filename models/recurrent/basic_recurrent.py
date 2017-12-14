from models.model_base import DetektorModel
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from util.tensor_provider import TensorProvider
from util.learning_rate_utilities import linear_geometric_curve, primary_secondary_plot
from util.utilities import save_fig, empty_folder
from pathlib import Path


class BasicRecurrent(DetektorModel):
    def __init__(self, recurrent_units=100, linear_units=(50,),
                 word_embedding=True, pos_tags=True, char_embedding=True,
                 n_batches=1000, batch_size=200,
                 display_step=10, results_path=None,
                 optimizer_class=tf.train.RMSPropOptimizer):
        """
        :param list | tuple linear_units: Number of units in fully-connected layers.
        :param int recurrent_units: number of units in recurrent layer.
        :param bool word_embedding: Use word-embeddings as inputs for network.
        :param bool pos_tags: Use pos-tags as inputs for network.
        :param bool char_embedding: Use character-embeddings as inputs for network.
        :param int n_batches:
        :param int batch_size:
        :param int display_step:
        """
        super().__init__(results_path)

        # For training
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.display_step = display_step

        # Settings
        self.use_char_embedding = char_embedding
        self.use_pos_tags = pos_tags
        self.use_word_embedding = word_embedding
        self.linear_units = linear_units
        self.recurrent_units = recurrent_units
        self.optimizer_class = optimizer_class

        # Uninitialized fields
        self.num_features = self.inputs = self.input_lengths = self.truth = self._rec_cell = self.rec_cell_outputs = \
            self.rec_cell_state = self.feedforward_activations = self._ffout_m = self._ffout_b = self._ffout_prod = \
            self._ffout_a = self.prediction = self.cost = self.learning_rate = self.optimize_op = \
            self._summary_merged = self._summary_train_writer = self.optimizer = None

    def initialize_model(self, tensor_provider):
        # Use model's graph
        with self._tf_graph.as_default():

            # Get number of features
            self.num_features = tensor_provider.input_dimensions(word_embedding=self.use_word_embedding,
                                                                 pos_tags=self.use_pos_tags,
                                                                 char_embedding=self.use_char_embedding)

            # Model inputs
            self.inputs = tf.placeholder(tf.float32, shape=[None, None, self.num_features], name='input')
            self.input_lengths = tf.placeholder(tf.int32, shape=[None], name='input_length')
            self.truth = tf.placeholder(tf.float32, [None, ])

            # Recurrent layer
            with tf.name_scope("recurrent_layer"):
                self._rec_cell = tf.nn.rnn_cell.GRUCell(num_units=self.recurrent_units)
                self.rec_cell_outputs, self.rec_cell_state = tf.nn.dynamic_rnn(cell=self._rec_cell,
                                                                               inputs=self.inputs,
                                                                               sequence_length=self.input_lengths,
                                                                               dtype=tf.float32)

            # Fully connected layers
            self.feedforward_activations = []
            last_units = self.recurrent_units
            c_input = self.rec_cell_state
            for layer_nr, n_units in enumerate(self.linear_units):
                layer_nr += 1
                with tf.name_scope("ff_layer{}".format(layer_nr)):
                    feedforward_weights = tf.Variable(tf.truncated_normal([last_units, n_units],
                                                                          stddev=np.sqrt(n_units)),
                                                      name="ff{}_m".format(layer_nr))
                    feedforward_bias = tf.Variable(tf.truncated_normal([n_units], stddev=np.sqrt(n_units)),
                                                   name="ff{}_b".format(layer_nr))
                    feedforward_product = tf.matmul(c_input, feedforward_weights)
                    feedforward_sum = feedforward_product + feedforward_bias
                    self.feedforward_activations.append(
                        tf.nn.relu(feedforward_sum, name="ff{}_activation".format(layer_nr))
                    )

                # Next layer
                c_input = self.feedforward_activations[-1]
                last_units = n_units

            # Output layer
            with tf.name_scope("output_layer"):
                self._ffout_m = tf.Variable(tf.truncated_normal([self.linear_units[-1], 1],
                                                                stddev=np.sqrt(self.linear_units[-1])),
                                            name="ffout_m")
                self._ffout_b = tf.Variable(tf.truncated_normal([1], stddev=1),
                                            name="ffout_b")
                self._ffout_prod = tf.matmul(self.feedforward_activations[-1], self._ffout_m)
                self._ffout_a = tf.transpose(self._ffout_prod + self._ffout_b)
                self.prediction = tf.nn.softmax(self._ffout_a, name="output")

            # Cost-function
            with tf.name_scope("Training"):
                with tf.name_scope("Cost"):
                    self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.truth,
                                                                                       logits=self._ffout_a))
                    tf.summary.scalar('cost', self.cost)

                # Gradient Descent
                with tf.name_scope("Optimizer"):
                    self.learning_rate = tf.placeholder(shape=(), dtype=tf.float32, name="learning_rate")
                    self.optimizer = self.optimizer_class(self.learning_rate)
                    self.optimize_op = self.optimizer.minimize(self.cost)

                # Merge summaries
                self._summary_merged = self._summary_train_writer = None
                if self.results_path is not None:
                    self._summary_merged = tf.summary.merge_all()
                    tensorboard_path = Path(self.results_path, "tensorboard_train")
                    empty_folder(tensorboard_path)
                    self._summary_train_writer = tf.summary.FileWriter(str(tensorboard_path), self._sess.graph)

    def fit(self, tensor_provider, train_idx, verbose=0):
        """
        :param TensorProvider tensor_provider:
        :param list train_idx:
        :param int verbose:
        :return:
        """

        # Use model's graph and run initializer
        with self._tf_graph.as_default():
            self._sess.run(tf.global_variables_initializer())

        # Close all figures and make a new one
        fig = None
        if self.results_path is not None:
            plt.close("all")
            plt.ioff()
            print("Making figure")
            fig = plt.figure(figsize=(14, 11))

        if verbose:
            print(verbose * " " + "Fitting {}".format(self.name()))
            verbose += 2

        # Get training data
        input_tensor = tensor_provider.load_concat_input_tensors(data_keys_or_idx=train_idx,
                                                                 word_embedding=self.use_word_embedding,
                                                                 char_embedding=self.use_char_embedding,
                                                                 pos_tags=self.use_pos_tags)
        output_truth = tensor_provider.load_labels(data_keys_or_idx=train_idx)
        input_lengths = tensor_provider.load_data_tensors(data_keys_or_idx=train_idx, word_counts=True)["word_counts"]
        train_idx = list(range(len(train_idx)))

        # Make learning rates
        learning_rates = linear_geometric_curve(n=self.n_batches,
                                                starting_value=1e-7,
                                                end_value=1e-18,
                                                geometric_component=3. / 4,
                                                geometric_end=5)

        # Calc sample probability based on class-size
        # TODO: Move this to own function and implement a "batch_strategy" input
        non_claim_if = 1.0 / sum(output_truth == 0)
        claim_if = 1.0 / sum(output_truth == 1)
        sample_weights = np.empty((len(train_idx)))
        sample_weights[output_truth == 0] = non_claim_if
        sample_weights[output_truth == 1] = claim_if
        sample_weights = sample_weights / sum(sample_weights)  # normalize to yield probabilities

        # Run training batches
        costs = []
        batches = []
        for batch_nr in range(self.n_batches):
            c_indices = np.random.choice(train_idx, self.batch_size, replace=False,
                                         p=sample_weights)
            c_inputs = input_tensor[c_indices, :, :]
            c_truth = output_truth[c_indices]
            c_input_lengths = input_lengths[c_indices]

            # Feeds
            feed_dict = {
                self.inputs: c_inputs,
                self.input_lengths: c_input_lengths,
                self.truth: c_truth,
                self.learning_rate: learning_rates[batch_nr]
            }

            # Fetching
            fetch = [self.optimize_op, self.cost]
            if self.results_path is not None:
                fetch.append(self._summary_merged)

            # Run batch training
            _, c, *summary = self._sess.run(fetches=fetch, feed_dict=feed_dict)

            # Tensorboard summaries
            if self.results_path is not None:
                self._summary_train_writer.add_summary(summary[0], batch_nr)

            # Note performance
            costs.append(c)
            batches.append(batch_nr + 1)

            if verbose:
                # Plot error and learning rate
                if self.results_path is not None:
                    fig.clear()
                    primary_secondary_plot(
                        primary_xs=batches,
                        primary_values=costs,
                        secondary_plots=[learning_rates],
                        x_limit=self.n_batches,
                        primary_label="Cost",
                        secondary_label="Learning Rate",
                        x_label="Batch",
                        title="BasicRecurrent: Cost and learning rate"
                    )
                    save_fig(Path(self.results_path, "training_curve"), only_pdf=True)

                # Print validation
                if (batch_nr + 1) % self.display_step == 0 and verbose:
                    print(verbose * " ", end="")
                    print("Batch {: 8d} / {: 8d}. cost = {:10.2f}. learning_rate = {:.2e}"
                          .format(batch_nr + 1,
                                  self.n_batches,
                                  c,
                                  learning_rates[batch_nr]))

        # Done
        if self.results_path is not None:
            plt.close("all")
            plt.ion()

    @classmethod
    def name(cls):
        return "BasicRecurrent"

    def predict(self, tensor_provider, predict_idx, additional_fetch=None):
        # Get data
        input_tensor = tensor_provider.load_concat_input_tensors(data_keys_or_idx=predict_idx,
                                                                 word_embedding=self.use_word_embedding,
                                                                 char_embedding=self.use_char_embedding,
                                                                 pos_tags=self.use_pos_tags)
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

        # Binary conversion
        binary_predictions = predictions > 0.5

        return predictions, binary_predictions
