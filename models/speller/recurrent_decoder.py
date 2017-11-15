import random
from collections import namedtuple

import numpy as np
import tensorflow as tf

from models.speller.utility import decoder


class RecurrentDecoder:
    def __init__(self, n_outputs, enc_state,
                 character_embedding, n_decoding_cells=None, output_encoding=None,
                 decoder_name="decoder"):
        self.n_decoding_cells = n_decoding_cells
        self.decoder_name = decoder_name
        self.character_embedding = character_embedding
        self._batch_queue = None

        # Placeholders
        self.targets_in = tf.placeholder(tf.int32, shape=[None, None], name='targets_in')
        self.targets_out = tf.placeholder(tf.int32, shape=[None, None], name='targets_out')
        self.targets_lengths = tf.placeholder(tf.int32, shape=[None], name='targets_length')
        self.targets_mask = tf.placeholder(tf.float32, shape=[None, None], name='t_mask')
        self.max_targets_length = tf.reduce_max(self.targets_lengths, name="max_targets_length")

        # Default input-encoder is one-hot
        if output_encoding is None:
            self._input_encoding = tf.constant(value=np.eye(n_outputs, n_outputs),
                                               dtype=tf.float32,
                                               shape=(n_outputs, n_outputs),
                                               name="onehot_encoding",
                                               verify_shape=True)
        else:
            self._input_encoding = output_encoding

        # Decoder
        with tf.variable_scope(self.decoder_name):
            self._targets_in_onehot = tf.gather(self._input_encoding, self.targets_in)
            self._targets_out_onehot = tf.gather(self._input_encoding, self.targets_out)

            if n_decoding_cells is not None:
                W_out = tf.get_variable('W_out', [n_decoding_cells, n_outputs])
                b_out = tf.get_variable('b_out', [n_outputs])
                dec_out, valid_dec_out = decoder(initial_state=enc_state,
                                                 target_input=self._targets_in_onehot,
                                                 target_len=self.targets_lengths,
                                                 num_units=n_decoding_cells,
                                                 embeddings=self._input_encoding,
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
                b_size = tf.shape(self.targets_in)[0]  # use a variable we know has batch_size in [0] TODO: may be wrong
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

    def _get_variables(self, training_scope):
        variables = tf.global_variables()
        variables = [var for var in variables
                     if self.decoder_name in var.name and training_scope not in var.name]
        # TODO: This string-handling may be slightly hardcoded
        variable_names = [var.name.split("/")[-1].split(":")[0] for var in variables]
        decoder_variables = [(name, var) for name, var in zip(variable_names, variables)]

        return decoder_variables

    def _create_saver(self, training_scope="training"):
        decoder_variables = self._get_variables(training_scope=training_scope)
        decoder_saver = tf.train.Saver(var_list=dict(decoder_variables))
        return decoder_saver

    def save(self, sess, file_path):
        # Get saver
        saver = self._create_saver()

        # Save
        saver.save(sess=sess, save_path=str(file_path))

    def load(self, sess, file_path, training_scope="training", do_check=False):
        variables_before = variables = None

        # Get saver
        with tf.name_scope(name=self.decoder_name):
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
            print("Loaded decoder, where {} / {} variables changed significantly ({:.2%})".format(check_sum,
                                                                                                  n_checks,
                                                                                                  check_mean))
