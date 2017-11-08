import random
from collections import namedtuple

import numpy as np
import tensorflow as tf

from models.speller.utility import decoder


class RecurrentSpeller:
    spelling_batch = namedtuple(typename="SpellingBatch",
                                field_names=[
                                    "word_sample",
                                    "input_lengths",
                                    "target_lengths",
                                    "inputs_numerical",
                                    "targets_in_numerical",
                                    "targets_out_numerical",
                                    "targets_mask"
                                ])

    def __init__(self, n_inputs, n_outputs, n_encoding_cells,
                 character_embedding, n_decoding_cells=None,
                 encoder_name="encoder", decoder_name="decoder", end_of_word_char="#"):
        self.end_of_word_char = end_of_word_char
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_encoding_cells = n_encoding_cells
        self.n_decoding_cells = n_decoding_cells
        self.encoder_name = encoder_name
        self.decoder_name = decoder_name
        self.character_embedding = character_embedding
        self._batch_queue = None

        # Placeholders
        self.inputs = tf.placeholder(tf.int32, shape=[None, None], name='input')
        self.input_lengths = tf.placeholder(tf.int32, shape=[None], name='input_length')
        self.targets_in = tf.placeholder(tf.int32, shape=[None, None], name='targets_in')
        self.targets_out = tf.placeholder(tf.int32, shape=[None, None], name='targets_out')
        self.targets_lengths = tf.placeholder(tf.int32, shape=[None], name='targets_length')
        self.targets_mask = tf.placeholder(tf.float32, shape=[None, None], name='t_mask')
        self.max_targets_length = tf.reduce_max(self.targets_lengths, name="max_targets_length")

        with tf.variable_scope(self.encoder_name):
            # One-hot encodings
            self._onehot_encoding = tf.constant(value=np.eye(n_inputs, n_inputs),
                                                dtype=tf.float32,
                                                shape=(n_inputs, n_inputs),
                                                name="onehot_encoding",
                                                verify_shape=True)
            self.inputs_onehot = tf.gather(self._onehot_encoding, self.inputs)
            self._targets_in_onehot = tf.gather(self._onehot_encoding, self.targets_in)
            self._targets_out_onehot = tf.gather(self._onehot_encoding, self.targets_out)

            # Forward encoding only
            self._spelling_enc_cell = tf.nn.rnn_cell.GRUCell(self.n_encoding_cells,
                                                             activation=None,
                                                             kernel_initializer=None,
                                                             bias_initializer=None)
            self._enc_outputs, self.enc_state = tf.nn.dynamic_rnn(cell=self._spelling_enc_cell,
                                                                  inputs=self.inputs_onehot,
                                                                  sequence_length=self.input_lengths,
                                                                  dtype=tf.float32)

        # Decoder
        with tf.variable_scope(self.decoder_name):
            if n_decoding_cells is not None:
                W_out = tf.get_variable('W_out', [n_decoding_cells, n_outputs])
                b_out = tf.get_variable('b_out', [n_outputs])
                dec_out, valid_dec_out = decoder(initial_state=self.enc_state,
                                                 target_input=self._targets_in_onehot,
                                                 target_len=self.targets_lengths,
                                                 num_units=n_decoding_cells,
                                                 embeddings=self._onehot_encoding,
                                                 W_out=W_out,
                                                 b_out=b_out)

                ######################################################

                # Reshaping for matmul (does not have broadcasting)
                # out_tensor: [batch_size * sequence_length, n_decoding_cells]
                out_tensor = tf.reshape(dec_out, [-1, n_decoding_cells])
                valid_out_tensor = tf.reshape(valid_dec_out, [-1, n_decoding_cells])

                # Computing logits
                out_tensor = tf.matmul(out_tensor, W_out) + b_out
                valid_out_tensor = tf.matmul(valid_out_tensor, W_out) + b_out

                # Determine shape of sequence format
                # TODO: Can this not be done with the given variables? (this is tedious)
                # TODO: My guess is [batch_size, None, n_decoding_cells]
                b_size = tf.shape(self.input_lengths)[0]  # use a variable we know has batch_size in [0]
                seq_len = tf.shape(self._targets_in_onehot)[1]  # variable we know has sequence length in [1]
                num_out = tf.constant(n_outputs)  # casting NUM_OUTPUTS to a tensor variable
                out_shape = tf.concat([tf.expand_dims(b_size, 0),
                                       tf.expand_dims(seq_len, 0),
                                       tf.expand_dims(num_out, 0)],
                                      axis=0)
                # out_shape = tf.concat([tf.expand_dims(b_size, 0),
                #                        tf.expand_dims(max_targets_length, 0),
                #                        tf.expand_dims(n_outputs, 0)],
                #                       axis=0)
                # out_shape = (batch_size, max_targets_length, n_outputs)

                # Reshaping back to sequence format
                # out_tensor: [batch_size, sequence_length, n_decoding_cells]
                out_tensor = tf.reshape(out_tensor, out_shape)
                valid_out_tensor = tf.reshape(valid_out_tensor, out_shape)

                # Set as prediction variables
                self.y = out_tensor
                self.y_valid = valid_out_tensor

            else:
                self.y = None
                self.y_valid = None

    def _get_encoder_variables(self, training_scope):
        variables = tf.global_variables()
        variables = [var for var in variables
                     if self.encoder_name in var.name and training_scope not in var.name]
        variable_names = [var.name.split(":")[0] for var in variables]
        encoder_variables = [(name, var) for name, var in zip(variable_names, variables)]

        return encoder_variables

    def _get_decoder_variables(self, training_scope):
        variables = tf.global_variables()
        variables = [var for var in variables
                     if self.decoder_name in var.name and training_scope not in var.name]
        variable_names = [var.name.split("/")[-1].split(":")[0] for var in variables]
        decoder_variables = [(name, var) for name, var in zip(variable_names, variables)]

        return decoder_variables

    def _create_encoder_saver(self, training_scope="training"):
        # TODO: This string-handling may be slightly hardcoded
        encoder_variables = self._get_encoder_variables(training_scope=training_scope)
        encoder_saver = tf.train.Saver(var_list=dict(encoder_variables))
        return encoder_saver

    def _create_decoder_saver(self, training_scope="training"):
        # TODO: This string-handling may be slightly hardcoded
        decoder_variables = self._get_decoder_variables(training_scope=training_scope)
        decoder_saver = tf.train.Saver(var_list=dict(decoder_variables))
        return decoder_saver

    def save_encoder(self, sess, file_path):
        # Get saver
        with tf.name_scope(name=self.encoder_name):
            saver = self._create_encoder_saver()

        # Save
        saver.save(sess=sess, save_path=str(file_path))

    def save_decoder(self, sess, file_path):
        # Get saver
        saver = self._create_decoder_saver()

        # Save
        saver.save(sess=sess, save_path=str(file_path))

    def load_encoder(self, sess, file_path, training_scope="training", do_check=False):
        variables_before = variables = None

        # Get saver
        with tf.name_scope(name=self.encoder_name):
            saver = self._create_encoder_saver()

        # Prepare check
        if do_check:
            variables = self._get_encoder_variables(training_scope=training_scope)
            variables_before = [var[1].eval(sess) for var in variables]

        # Restore
        saver.restore(sess, str(file_path))

        # Finish check
        if variables is not None:
            variables_after = [var[1].eval(sess) for var in variables]
            checks = [~np.isclose(before, after).flatten()
                      for before, after in zip(variables_before, variables_after)]
            concat_checks = np.concatenate(checks)
            n_checks = concat_checks.shape[0]
            check_sum = np.sum(concat_checks)
            check_mean = np.mean(concat_checks)
            print("Loaded encoder, where {} / {} variables changed significantly ({:.2%})".format(check_sum,
                                                                                                  n_checks,
                                                                                                  check_mean))
            if check_mean > 0.01:
                for var, check, before, after in zip(variables, checks, variables_before, variables_after):
                    before_str = "Before({})".format(self._short_stat(before))
                    after_str = "After({})".format(self._short_stat(after))
                    print("   {:50s} : {:6d} / {:6d} changed. {:70s}. {:70s}".format(var[1].name,
                                                                                     sum(check),
                                                                                     len(check),
                                                                                     before_str,
                                                                                     after_str))

    def _short_stat(self, matrix):
        """
        :param np.ndarray matrix:
        :return:
        """
        mean = matrix.mean()
        c_max = matrix.max()
        c_min = matrix.min()
        var = matrix.var()
        return "mean {:.2e}, var {:.2e}, min {:.2e}, max {:.2e}".format(mean, var, c_min, c_max)

    def load_decoder(self, sess, file_path, training_scope="training", do_check=False):
        variables_before = variables = None

        # Get saver
        with tf.name_scope(name=self.decoder_name):
            saver = self._create_decoder_saver()

        # Prepare check
        if do_check:
            variables = self._get_decoder_variables(training_scope=training_scope)
            variables_before = [var[1].eval(sess) for var in variables]

        # Restore
        saver.restore(sess, str(file_path))

        # Finish check
        if variables is not None:
            variables_after = [var[1].eval(sess) for var in variables]
            checks = [~np.isclose(before, after).flatten()
                      for before, after in zip(variables_before, variables_after)]
            concat_checks = np.concatenate(checks)
            n_checks = concat_checks.shape[0]
            check_sum = np.sum(concat_checks)
            check_mean = np.mean(concat_checks)
            print("Loaded decoder, where {} / {} variables changed significantly ({:.2%})".format(check_sum,
                                                                                                  n_checks,
                                                                                                  check_mean))
            if check_mean < 0.99:
                for var, check in zip(variables, checks):
                    print("   {} : {} / {}".format(var[1].name, sum(check), len(check)))

    def str2code(self, string):
        return [self.character_embedding[val] for val in string]

    def code2str(self, code):
        return "".join([self.character_embedding[val] for val in code])

    def next_batch(self, words, batch_size):
        # Initialize randomized order of words
        if self._batch_queue is None:
            indices = list(range(len(words)))
            random.shuffle(indices)
            self._batch_queue = indices

        # Reset randomized cycle
        elif len(self._batch_queue) < batch_size:
            indices = list(range(len(words)))
            random.shuffle(indices)
            self._batch_queue = indices + self._batch_queue

        # Grab sample
        self._batch_queue, sample_indices = self._batch_queue[:-batch_size], self._batch_queue[-batch_size:]
        word_sample = [words[idx] for idx in sample_indices]

        # Produce and return batch
        return self.process_batch(word_sample=word_sample)

    def process_batch(self, word_sample):

        # Determine lengths of inputs and targets
        input_lengths = [len(a_word) for a_word in word_sample]
        target_lengths = [val + 1 for val in input_lengths]
        max_word_length = max(input_lengths)

        # Encode inputs into numerical. Pad with spaces before coding.
        inputs_numerical = np.array([self.str2code(a_word + " " * (max_word_length - len(a_word)))
                                     for a_word in word_sample])

        # Encode ingoing targets into numerical. Add end-of-line and pad with spaces before coding.
        targets_in_numerical = np.array([self.str2code(self.end_of_word_char + a_word +
                                                       " " * (max_word_length - len(a_word)))
                                         for a_word in word_sample])

        # Encode outgoing targets into numerical. Pad with end-of-line and spaces before coding.
        targets_out_numerical = np.array([self.str2code(a_word + self.end_of_word_char +
                                                        " " * (max_word_length - len(a_word)))
                                          for a_word in word_sample])

        # Create mask for locating targets
        targets_mask = np.array([[1] * (len(a_word) + 1) +  # Targets are at the word
                                 [0] * (max_word_length - len(a_word))  # The rest is padding and not target
                                 for a_word in word_sample])

        # Return
        return self.spelling_batch(word_sample,
                                   input_lengths,
                                   target_lengths,
                                   inputs_numerical,
                                   targets_in_numerical,
                                   targets_out_numerical,
                                   targets_mask)

    def get_encoding(self, sess, words):
        # Process words into inputs
        (c_word_sample, c_input_lengths, c_target_lengths, c_inputs_numerical, c_targets_in_numerical,
         c_targets_out_numerical, c_targets_mask) = self.process_batch(word_sample=words)

        # Create feed-dictionary
        feed_dict = {self.inputs: c_inputs_numerical,
                     self.input_lengths: c_input_lengths,
                     self.targets_in: c_targets_in_numerical,
                     self.targets_out: c_targets_out_numerical,
                     self.targets_lengths: c_target_lengths,
                     self.targets_mask: c_targets_mask,
                     }

        # Forward pass
        fetches = [self.enc_state]
        res = sess.run(fetches=fetches, feed_dict=feed_dict)

        return res[0]

    def forward_pass(self, sess, words, verbose=False):
        """
        Forward pass of the recurrent speller.
        :param tf.Session sess:
        :param list[str] words:
        :param bool verbose:
        :return:
        """
        # Process words into inputs
        (c_word_sample, c_input_lengths, c_target_lengths, c_inputs_numerical, c_targets_in_numerical,
         c_targets_out_numerical, c_targets_mask) = self.process_batch(word_sample=words)

        if verbose:
            for i, word in enumerate(c_word_sample):
                ts_in = self.str2code(self.end_of_word_char + word)

                print("\nSAMPLE", i)
                print("Word:\t\t\t\t", word)
                print("Input code:\t\t\t", self.str2code(word + self.end_of_word_char))
                print("Target inputs:\t\t", ts_in)

        # Create feed-dictionary
        feed_dict = {self.inputs: c_inputs_numerical,
                     self.input_lengths: c_input_lengths,
                     self.targets_in: c_targets_in_numerical,
                     self.targets_out: c_targets_out_numerical,
                     self.targets_lengths: c_target_lengths,
                     self.targets_mask: c_targets_mask,
                     }

        # Forward pass
        fetches = [self.y]
        res = sess.run(fetches=fetches, feed_dict=feed_dict)
        y = res[0]
        if verbose:
            print("\ny", y.shape)

        # Test validation forward pass
        fetches = [self.y_valid]
        res = sess.run(fetches=fetches, feed_dict=feed_dict)
        y_valid = res[0]
        if verbose:
            print("y_valid", y_valid.shape)

        # Return
        return y, y_valid
