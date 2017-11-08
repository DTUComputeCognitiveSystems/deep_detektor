import numpy as np
import tensorflow as tf


class RecurrentEncoder:
    def __init__(self, n_inputs, n_encoding_cells,
                 character_embedding,
                 encoder_name="encoder"):
        self.n_inputs = n_inputs
        self.n_encoding_cells = n_encoding_cells
        self.encoder_name = encoder_name
        self.character_embedding = character_embedding
        self._batch_queue = None

        # Placeholders
        self.inputs = tf.placeholder(tf.int32, shape=[None, None], name='input')
        self.input_lengths = tf.placeholder(tf.int32, shape=[None], name='input_length')

        with tf.variable_scope(self.encoder_name):
            # One-hot encodings
            self._onehot_encoding = tf.constant(value=np.eye(n_inputs, n_inputs),
                                                dtype=tf.float32,
                                                shape=(n_inputs, n_inputs),
                                                name="onehot_encoding",
                                                verify_shape=True)
            self._inputs_onehot = tf.gather(self._onehot_encoding, self.inputs)

            # Forward encoding only
            self._spelling_enc_cell = tf.nn.rnn_cell.GRUCell(self.n_encoding_cells,
                                                             activation=None,
                                                             kernel_initializer=None,
                                                             bias_initializer=None)
            self.enc_outputs, self.enc_state = tf.nn.dynamic_rnn(cell=self._spelling_enc_cell,
                                                                 inputs=self._inputs_onehot,
                                                                 sequence_length=self.input_lengths,
                                                                 dtype=tf.float32)

    def _get_variables(self, training_scope):
        variables = tf.global_variables()
        variables = [var for var in variables
                     if self.encoder_name in var.name and training_scope not in var.name]
        variable_names = [var.name.split("/")[-1].split(":")[0] for var in variables]
        encoder_variables = [(name, var) for name, var in zip(variable_names, variables)]

        return encoder_variables

    def _create_saver(self, training_scope="training"):
        encoder_variables = self._get_variables(training_scope=training_scope)
        encoder_saver = tf.train.Saver(var_list=dict(encoder_variables))
        return encoder_saver

    def save(self, sess, file_path):
        # Get saver
        with tf.name_scope(name=self.encoder_name):
            saver = self._create_saver()

        # Save
        saver.save(sess=sess, save_path=str(file_path))

    def load(self, sess, file_path, training_scope="training", do_check=False):
        variables_before = variables = None

        # Get saver
        with tf.name_scope(name=self.encoder_name):
            saver = self._create_saver()

        # Prepare check
        if do_check:
            variables = self._get_variables(training_scope=training_scope)
            variables_before = [var[1].eval(sess) for var in variables]

        # Restore
        saver.restore(sess, str(file_path))

        # Finish check
        if variables is not None:
            variables_after = [var[1].eval(sess) for var in variables]
            checks = [~np.isclose(before, after).flatten()
                      for before, after in zip(variables_before, variables_after)]
            checks = np.concatenate(checks)
            n_checks = checks.shape[0]
            check_sum = np.sum(checks)
            check_mean = np.mean(checks)
            print("Loaded encoder, where {} / {} variables changed significantly ({:.2%})".format(check_sum,
                                                                                                  n_checks,
                                                                                                  check_mean))

    def str2code(self, string):
        return [self.character_embedding[val] for val in string]

    def code2str(self, code):
        return "".join([self.character_embedding[val] for val in code])
