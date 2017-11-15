import tensorflow as tf
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import tensor_array_ops


def sequence_loss_tensor(logits, targets, num_classes, weights=None,
                         average_across_timesteps=False,
                         softmax_loss_function=None, name="sequenceLoss"):
    """
    Weighted cross-entropy loss for a sequence of logits (per example).
    It is a modification of TensorFlow's own sequence_to_sequence_loss.
    TensorFlow's seq2seq loss works with a 2D list instead of a 3D tensors.

    :param tf.Tensor logits: Logits for each class for all samples. [batch_size, (sequence_length), num_classes]
    :param tf.Tensor targets: True classes of samples. [batch_size, (sequence_length)]
    :param int | tf.Tensor num_classes: The total number of classes.
    :param tf.Tensor weights: Weighing of each sample. [batch_size, (sequence_length)]
    :param bool average_across_timesteps: Average loss across time-dimension.
    :param Callable softmax_loss_function: Method used for computing loss.
    :param str name: Name of loss-functions scope.
    :return: tf.Tensor
    """
    if average_across_timesteps:
        raise NotImplementedError("Averaging across time-steps has not been implemented yet. ")

    with tf.variable_scope(name):
        # Flatten logits for using softmax operation, and targets for comparison
        # logits_flat: [batch_size * (sequence_length), num_classes]
        # targets: [batch_size * (sequence_length)]
        logits_flat = tf.reshape(logits, [-1, num_classes])
        targets = tf.reshape(targets, [-1])

        # If a custom loss function is given, then use that. Otherwise default
        # cross_ent: [batch_size * (sequence_length)]
        if softmax_loss_function is None:
            cross_ent = nn_ops.sparse_softmax_cross_entropy_with_logits(logits=logits_flat, labels=targets)
        else:
            cross_ent = softmax_loss_function(logits_flat, targets)

        # Weigh cross-entropy if wanted
        if weights is not None:
            cross_ent = cross_ent * tf.reshape(weights, [-1])

        # Cross-entropy sum
        cross_ent = tf.reduce_sum(cross_ent)

        # Divide by total weighting
        # TODO: Couldn't you just normalize the weights first?
        if weights is not None:
            total_size = tf.reduce_sum(weights)
            total_size += 1e-12  # to avoid division by zero
            cross_ent /= total_size

        return cross_ent


def seq_loss_accuracy(logits, targets_out, targets_mask, n_outputs, regularization_scale=None):
    # Get loss using weighing for selecting targets at mask
    loss = sequence_loss_tensor(logits=logits,
                                targets=targets_out,
                                weights=targets_mask,
                                num_classes=n_outputs)  # notice that we use ts_out here!

    # Add regularization
    if regularization_scale is not None:
        # Define regularization scheme
        regularize = tf.contrib.layers.l2_regularizer(regularization_scale)

        # Add regularization to all trainable variables
        # TODO: Make the speller capable of returning specific weights for regularization (as DNC)
        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        # Add sum of regularization
        reg_term = sum([regularize(param) for param in params])
        loss += reg_term

    # Determine direct predictions
    predictions = tf.to_int32(tf.argmax(logits, axis=2))

    # Determine correct predictions and accuracy
    correct = tf.to_float(tf.equal(predictions, targets_out)) * targets_mask
    accuracy = tf.reduce_sum(correct) / tf.reduce_sum(targets_mask)

    return loss, accuracy, predictions


class TFPrinter:
    @staticmethod
    def _print_collection(key, scope, indentation):
        # Get names and shapes
        var_names, var_shapes = zip(*[(var.name, var.value().get_shape())
                                      for var in tf.get_collection(key=key, scope=scope)])

        # Determine formatter from longest name
        max_name_length = max([len(name) for name in var_names])
        formatter = " " * indentation + "{{:{}s}}  :  {{}}".format(max_name_length)

        # Print names and shapes
        for name, shape in zip(var_names, var_shapes):
            print(formatter.format(name, shape))

    @staticmethod
    def activations(scope=None, indentation=0):
        return TFPrinter._print_collection(tf.GraphKeys.ACTIVATIONS, scope, indentation)

    @staticmethod
    def asset_filepaths(scope=None, indentation=0):
        return TFPrinter._print_collection(tf.GraphKeys.ASSET_FILEPATHS, scope, indentation)

    @staticmethod
    def biases(scope=None, indentation=0):
        return TFPrinter._print_collection(tf.GraphKeys.BIASES, scope, indentation)

    @staticmethod
    def concatenated_variables(scope=None, indentation=0):
        return TFPrinter._print_collection(tf.GraphKeys.CONCATENATED_VARIABLES, scope, indentation)

    @staticmethod
    def cond_context(scope=None, indentation=0):
        return TFPrinter._print_collection(tf.GraphKeys.COND_CONTEXT, scope, indentation)

    @staticmethod
    def eval_step(scope=None, indentation=0):
        return TFPrinter._print_collection(tf.GraphKeys.EVAL_STEP, scope, indentation)

    @staticmethod
    def global_step(scope=None, indentation=0):
        return TFPrinter._print_collection(tf.GraphKeys.GLOBAL_STEP, scope, indentation)

    @staticmethod
    def global_variables(scope=None, indentation=0):
        return TFPrinter._print_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope, indentation)

    @staticmethod
    def init_op(scope=None, indentation=0):
        return TFPrinter._print_collection(tf.GraphKeys.INIT_OP, scope, indentation)

    @staticmethod
    def local_init_op(scope=None, indentation=0):
        return TFPrinter._print_collection(tf.GraphKeys.LOCAL_INIT_OP, scope, indentation)

    @staticmethod
    def local_resources(scope=None, indentation=0):
        return TFPrinter._print_collection(tf.GraphKeys.LOCAL_RESOURCES, scope, indentation)

    @staticmethod
    def local_variables(scope=None, indentation=0):
        return TFPrinter._print_collection(tf.GraphKeys.LOCAL_VARIABLES, scope, indentation)

    @staticmethod
    def losses(scope=None, indentation=0):
        return TFPrinter._print_collection(tf.GraphKeys.LOSSES, scope, indentation)

    @staticmethod
    def model_variables(scope=None, indentation=0):
        return TFPrinter._print_collection(tf.GraphKeys.MODEL_VARIABLES, scope, indentation)

    @staticmethod
    def moving_average_variables(scope=None, indentation=0):
        return TFPrinter._print_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES, scope, indentation)

    @staticmethod
    def queue_runners(scope=None, indentation=0):
        return TFPrinter._print_collection(tf.GraphKeys.QUEUE_RUNNERS, scope, indentation)

    @staticmethod
    def ready_for_local_init_op(scope=None, indentation=0):
        return TFPrinter._print_collection(tf.GraphKeys.READY_FOR_LOCAL_INIT_OP, scope, indentation)

    @staticmethod
    def ready_op(scope=None, indentation=0):
        return TFPrinter._print_collection(tf.GraphKeys.READY_OP, scope, indentation)

    @staticmethod
    def regularization_losses(scope=None, indentation=0):
        return TFPrinter._print_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope, indentation)

    @staticmethod
    def resources(scope=None, indentation=0):
        return TFPrinter._print_collection(tf.GraphKeys.RESOURCES, scope, indentation)

    @staticmethod
    def saveable_objects(scope=None, indentation=0):
        return TFPrinter._print_collection(tf.GraphKeys.SAVEABLE_OBJECTS, scope, indentation)

    @staticmethod
    def savers(scope=None, indentation=0):
        return TFPrinter._print_collection(tf.GraphKeys.SAVERS, scope, indentation)

    @staticmethod
    def summaries(scope=None, indentation=0):
        return TFPrinter._print_collection(tf.GraphKeys.SUMMARIES, scope, indentation)

    @staticmethod
    def summary_op(scope=None, indentation=0):
        return TFPrinter._print_collection(tf.GraphKeys.SUMMARY_OP, scope, indentation)

    @staticmethod
    def table_initializers(scope=None, indentation=0):
        return TFPrinter._print_collection(tf.GraphKeys.TABLE_INITIALIZERS, scope, indentation)

    @staticmethod
    def trainable_resource_variables(scope=None, indentation=0):
        return TFPrinter._print_collection(tf.GraphKeys.TRAINABLE_RESOURCE_VARIABLES, scope, indentation)

    @staticmethod
    def trainable_variables(scope=None, indentation=0):
        return TFPrinter._print_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope, indentation)

    @staticmethod
    def train_op(scope=None, indentation=0):
        return TFPrinter._print_collection(tf.GraphKeys.TRAIN_OP, scope, indentation)

    @staticmethod
    def update_ops(scope=None, indentation=0):
        return TFPrinter._print_collection(tf.GraphKeys.UPDATE_OPS, scope, indentation)

    @staticmethod
    def variables(scope=None, indentation=0):
        return TFPrinter._print_collection(tf.GraphKeys.VARIABLES, scope, indentation)

    @staticmethod
    def weights(scope=None, indentation=0):
        return TFPrinter._print_collection(tf.GraphKeys.WEIGHTS, scope, indentation)

    @staticmethod
    def while_context(scope=None, indentation=0):
        return TFPrinter._print_collection(tf.GraphKeys.WHILE_CONTEXT, scope, indentation)


def print_tf_variables(indentation=0):
    # Get names and shapes
    var_names, var_shapes = zip(*[(var.name, var.value().get_shape())
                                  for var in tf.global_variables()])

    # Determine formatter from longest name
    max_name_length = max([len(name) for name in var_names])
    formatter = " " * indentation + "{{:{}s}}  :  {{}}".format(max_name_length)

    # Print names and shapes
    for name, shape in zip(var_names, var_shapes):
        print(formatter.format(name, shape))


def decoder(initial_state, target_input, target_len, num_units,
            embeddings, W_out, b_out,
            W_z_x_init=tf.truncated_normal_initializer(stddev=0.1),
            W_z_h_init=tf.truncated_normal_initializer(stddev=0.1),
            W_r_x_init=tf.truncated_normal_initializer(stddev=0.1),
            W_r_h_init=tf.truncated_normal_initializer(stddev=0.1),
            W_c_x_init=tf.truncated_normal_initializer(stddev=0.1),
            W_c_h_init=tf.truncated_normal_initializer(stddev=0.1),
            b_z_init=tf.constant_initializer(0.0),
            b_r_init=tf.constant_initializer(0.0),
            b_c_init=tf.constant_initializer(0.0),
            swap=False):
    """
    Imported from the Deep Learning Course.
    """

    # we need the max seq len to optimize our RNN computation later on
    max_sequence_length = tf.reduce_max(target_len)
    # target_dims is just the embedding size
    target_dims = target_input.get_shape()[2]
    # set up weights for the GRU gates
    var = tf.get_variable  # for ease of use
    # target_dims + num_units is because we stack embeddings and prev. hidden state to
    # optimize speed
    W_z_x = var('W_z_x', shape=[target_dims, num_units], initializer=W_z_x_init)
    W_z_h = var('W_z_h', shape=[num_units, num_units], initializer=W_z_h_init)
    b_z = var('b_z', shape=[num_units], initializer=b_z_init)
    W_r_x = var('W_r_x', shape=[target_dims, num_units], initializer=W_r_x_init)
    W_r_h = var('W_r_h', shape=[num_units, num_units], initializer=W_r_h_init)
    b_r = var('b_r', shape=[num_units], initializer=b_r_init)
    W_c_x = var('W_c_x', shape=[target_dims, num_units], initializer=W_c_x_init)
    W_c_h = var('W_c_h', shape=[num_units, num_units], initializer=W_c_h_init)
    b_c = var('b_h', shape=[num_units], initializer=b_c_init)

    # make inputs time-major
    inputs = tf.transpose(target_input, perm=[1, 0, 2])
    # make tensor array for inputs, these are dynamic and used in the while-loop
    # these are not in the api documentation yet, you will have to look at github.com/tensorflow
    input_ta = tensor_array_ops.TensorArray(tf.float32, size=1, dynamic_size=True)
    # input_ta = input_ta.unpack(inputs)
    input_ta = input_ta.unstack(inputs)

    # function to the while-loop, for early stopping
    def decoder_cond(time, state, output_ta_t):
        return tf.less(time, max_sequence_length)

    # the body_builder is just a wrapper to parse feedback
    def decoder_body_builder(feedback=False):
        # the decoder body, this is where the RNN magic happens!
        def decoder_body(time, old_state, output_ta_t):
            # when validating we need previous prediction, handle in feedback
            if feedback:
                def from_previous():
                    prev_1 = tf.matmul(old_state, W_out) + b_out
                    return tf.gather(embeddings, tf.argmax(prev_1, 1))

                x_t = tf.cond(tf.greater(time, 0), from_previous, lambda: input_ta.read(0))
            else:
                # else we just read the next timestep
                x_t = input_ta.read(time)

            # calculate the GRU
            z = tf.sigmoid(tf.matmul(x_t, W_z_x) + tf.matmul(old_state, W_z_h) + b_z)  # update gate
            r = tf.sigmoid(tf.matmul(x_t, W_r_x) + tf.matmul(old_state, W_r_h) + b_r)  # reset gate
            c = tf.tanh(tf.matmul(x_t, W_c_x) + tf.matmul(r * old_state, W_c_h) + b_c)  # proposed new state
            new_state = (1 - z) * c + z * old_state  # new state

            # writing output
            output_ta_t = output_ta_t.write(time, new_state)

            # return in "input-to-next-step" style
            return (time + 1, new_state, output_ta_t)

        return decoder_body

    # set up variables to loop with
    output_ta = tensor_array_ops.TensorArray(tf.float32, size=1, dynamic_size=True, infer_shape=False)
    time = tf.constant(0)
    loop_vars = [time, initial_state, output_ta]

    # run the while-loop for training
    _, state, output_ta = tf.while_loop(decoder_cond,
                                        decoder_body_builder(),
                                        loop_vars,
                                        swap_memory=swap)
    # run the while-loop for validation
    _, valid_state, valid_output_ta = tf.while_loop(decoder_cond,
                                                    decoder_body_builder(feedback=True),
                                                    loop_vars,
                                                    swap_memory=swap)
    # returning to batch major
    dec_out = tf.transpose(output_ta.stack(), perm=[1, 0, 2])
    valid_dec_out = tf.transpose(valid_output_ta.stack(), perm=[1, 0, 2])
    return dec_out, valid_dec_out
