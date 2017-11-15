"""
Arguments: [0-48]
"""
import csv
import json
import pickle
import random
import string
import warnings
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from models.speller.recurrent_speller import RecurrentSpeller
from models.speller.utility import seq_loss_accuracy, print_tf_variables, TFPrinter
from util.utilities import ensure_folder, save_fig

# from tensorflow.python.tools import inspect_checkpoint

plt.close("all")

# Script settings
restore_encoder = False
restore_decoder = False
overwrite_jobs = True

# Model settings
end_of_word_char = "$"

# Training settings
clip_norm = 1
training_batch_size = int(1e2)
val_batch_size = int(5e2)
test_batch_size = int(1e3)
n_batches = int(1e4)
val_interval = int(1e1)


#####


def random_sequence_batch(character_list, min_length, max_length, batch_size):
    character_list = [val for val in character_list if val != end_of_word_char]
    lengths = np.random.randint(low=min_length, high=max_length + 1, size=batch_size).tolist()

    sequences = []
    for length in lengths:
        sequence = "".join(np.random.choice(list(character_list), size=length, replace=True))
        sequences.append(sequence)

    return sequences


# Output directory
output_dir = Path("data", f"spelling_model")
ensure_folder(output_dir)

# Model settings
cells = 100


#####

print("Getting data.")

# Get data
texts = []
matrix_path = Path("data", "DeepFactData", "annotated", "data_matrix_sample_programs.csv")
with matrix_path.open("r") as file:
    csv_reader = csv.reader(file, delimiter=",")
    for row in csv_reader:
        texts.append(eval(row[4]).decode())

# Create word-vocabulary
vocabulary = set(
    word
    for sentence in texts
    for word in sentence.split()
)

# Character vocabulary
char_vocabulary_original = set(
    char
    for word in vocabulary
    for char in word
).union(str(string.ascii_lowercase))

# Keep sensible letters
char_vocabulary = set(val for val in char_vocabulary_original
                      if val in string.ascii_letters + string.digits + string.punctuation + "æøåÆØÅ")
questionable_letters = char_vocabulary_original.difference(char_vocabulary)
print("Ignoring the following letters: {}".format(questionable_letters))

# String translator (reducing number of characters)
digit_from = "".join([str(val) for val in range(10)])
digit_to = "#" * len(digit_from)
upper_from = string.ascii_uppercase + "ÆØÅ"
upper_to = "".join([val.lower() for val in upper_from])
string_translator = str.maketrans(digit_from + upper_from, digit_to + upper_to)

with Path(output_dir, "string_translator.json").open("w") as file:
    json.dump(string_translator, file)

# Input character vocabulary
input_char_vocabulary = set(val.translate(string_translator) for val in char_vocabulary)

# Words and word-lengths
all_words = list(sorted(vocabulary))
word_lengths = [len(word) for word in all_words]

# # Full vocabulary (before and after transformation)
# full_vocabulary = vocabulary.union("".join(char for char in val.translate(string_translator)
#                                            if char in input_char_vocabulary)
#                                    for val in vocabulary)

##########
# Character frequencies

# Count characters in english set
character_counts = Counter([val
                            for val in "".join(all_words).translate(string_translator)
                            if val in input_char_vocabulary])

# Final character_list
character_list = sorted(list(character_counts.keys()))
char_embedding = {val: idx for idx, val in enumerate(character_list + [" ", end_of_word_char])}

with Path(output_dir, "char_embedding.json").open("w") as file:
    json.dump(char_embedding, file)

# Compute probabilities (some characters are left out relative to histogram)
character_probabilities = [character_counts[val] for val in character_list]
character_probabilities = np.array(character_probabilities) / sum(character_probabilities)

##########
# Data set

print("Creating synthetic data.")
# Sample random sequences as words
words = []
for _ in range(20):
    for length in word_lengths:
        sequence = "".join(np.random.choice(character_list,
                                            size=length,
                                            replace=True,
                                            p=character_probabilities))
        words.append(sequence)

n_words = len(words)


##########

# Training, validation and test-set sizes
r_train = 0.98
r_val = 0.01

# Split words
random.shuffle(words)
n_train = int(n_words * r_train)
n_val = int(n_words * r_val)
n_test = n_words - n_train - n_val
words_train, words_val, words_test = words[:n_train], words[n_train:n_train + n_val], words[n_train + n_val:]

pickle.dump(words_train, Path(output_dir, "words_train.p").open("wb"))
pickle.dump(words_val, Path(output_dir, "words_val.p").open("wb"))
pickle.dump(words_test, Path(output_dir, "words_test.p").open("wb"))

print("Data sets:")
print("\t{:13s}: {: 10,d}".format("Training", len(words_train)))
print("\t{:13s}: {: 10,d}".format("Validation", len(words_val)))
print("\t{:13s}: {: 10,d}".format("Test", len(words_test)))


##########

# Resetting tensorflow graph
tf.reset_default_graph()

# Network setup
n_inputs = len(character_list) + 2  # Space for filling sequences # for end-of-line
n_outputs = n_inputs
n_encoding_cells = cells
n_decoding_cells = cells

######################################################
print("Defining Model")

# Model
recurrent_speller = RecurrentSpeller(n_inputs=n_inputs,
                                     n_outputs=n_outputs,
                                     n_encoding_cells=n_encoding_cells,
                                     n_decoding_cells=n_decoding_cells,
                                     character_embedding=char_embedding,
                                     end_of_word_char=end_of_word_char)

######################################################
print("Defining Training")

# Training
with tf.variable_scope("training"):
    learning_rate = tf.placeholder(shape=(), dtype=tf.float32, name="learning_rate")

    # Determine performance for validation and training
    loss, accuracy, predictions = seq_loss_accuracy(recurrent_speller.y,
                                                    targets_out=recurrent_speller.targets_out,
                                                    targets_mask=recurrent_speller.targets_mask,
                                                    n_outputs=n_outputs)
    loss_valid, accuracy_valid, predictions_valid = seq_loss_accuracy(recurrent_speller.y_valid,
                                                                      targets_out=recurrent_speller.targets_out,
                                                                      targets_mask=recurrent_speller.targets_mask,
                                                                      n_outputs=n_outputs)

    # Use global step to keep track of our iterations
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Pick optimizer, try momentum or adadelta
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # Extract gradients for each variable
    grads_and_vars = optimizer.compute_gradients(loss)

    # Clip gradients
    if clip_norm is not None:
        gradients, variables = zip(*grads_and_vars)
        clipped_gradients, global_norm = tf.clip_by_global_norm(gradients, clip_norm)
        grads_and_vars = zip(clipped_gradients, variables)

    # Apply gradients as training operator
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

######################################################
# Start the session and restore variables
print("Starting Session")

# Print all variables and shapes
print_tf_variables()

# Session
gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.35)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts))

# Initialize graph
sess.run(tf.global_variables_initializer())

# print_tensors_in_checkpoint_file(str(Path(output_dir, "checkpoint", "speller_encode.ckpt")),
#                                  tensor_name="",
#                                  all_tensors=True)

# Restore encoder
if restore_encoder:
    print("Restoring encoder from files.")
    recurrent_speller.load_encoder(sess=sess,
                                   file_path=Path(output_dir, "checkpoint", "speller_encode.ckpt"),
                                   do_check=True)

# Restore decoder
if restore_decoder:
    print("Restoring decoder from files.")
    recurrent_speller.load_decoder(sess=sess,
                                   file_path=Path(output_dir, "checkpoint", "speller_decode.ckpt"),
                                   do_check=True)

if not restore_encoder and not restore_decoder:
    print("No model was restored from files. ")

######################################################
# Test forward passes
print("Testing Forward Pass")

# Grab a batch
c_batch_size = 3

c_words = random.sample(words_train, c_batch_size)

_ = recurrent_speller.forward_pass(sess=sess,
                                   words=c_words,
                                   verbose=True)

######################################################

print("-" * 75)
print("Trainable Variables")
TFPrinter.trainable_variables()

plt.close("all")
print("\nTraining")

# Learning rate
# learning_rates = np.linspace(1e-1, 1e-5, samples_to_process + 1)
linear_component = int(n_batches * (3 / 3))
learning_rates = np.geomspace(2e-2, 1e-5, n_batches + 1)
learning_rates[:linear_component] += np.linspace(1e-2, 1e-15, linear_component)

# Plotting
plt.figure()
ax1 = plt.gca()  # type: plt.Axes
ax2 = plt.twinx(ax1)  # type: plt.Axes
ax2.plot(learning_rates, color="green", alpha=1.0 / 3)
ax2.set_xlim(0, n_batches)
ax2.set_ylabel("Learning Rate", color="green")
ylim = ax2.get_ylim()
ax2.set_ylim(0, ylim[1])
ax2.tick_params(axis='y', colors='green')
ax1.tick_params(axis='y', colors='blue')

val_batch_nr = []
costs, accs_val = [], []

# Training loop
batch_nr = -1
while batch_nr < n_batches:
    batch_nr += 1
    do_validation = batch_nr % val_interval == 0

    # Pick learning rate from curve
    c_learning_rate = learning_rates[batch_nr]

    if do_validation:
        print("{} / {}".format(batch_nr, n_batches), end="")

    # Pick out batch
    c_words = random.sample(words_train, training_batch_size)

    # Load data
    (c_word_sample, c_input_lengths, c_target_lengths, c_inputs_numerical, c_targets_in_numerical,
     c_targets_out_numerical, c_targets_mask) = recurrent_speller.process_batch(c_words)

    # Make fetches
    fetches_tr = [train_op, loss, accuracy]

    # Set up feed dict
    feed_dict_tr = {
        recurrent_speller.inputs: c_inputs_numerical,
        recurrent_speller.input_lengths: c_input_lengths,
        recurrent_speller.targets_in: c_targets_in_numerical,
        recurrent_speller.targets_out: c_targets_out_numerical,
        recurrent_speller.targets_lengths: c_target_lengths,
        recurrent_speller.targets_mask: c_targets_mask,
        learning_rate: c_learning_rate
    }

    # Run the model
    res = tuple(sess.run(fetches=fetches_tr, feed_dict=feed_dict_tr))
    _, batch_cost, batch_acc = res
    costs += [batch_cost]

    # Validation data
    if do_validation:
        val_batch_nr.append(batch_nr)

        # Process validation batch
        val_words = random.sample(words_val, val_batch_size)
        (c_word_sample_val, c_input_lengths_val, c_target_lengths_val, c_inputs_numerical_val,
         c_targets_in_numerical_val,
         c_targets_out_numerical_val, c_targets_mask_val) = recurrent_speller.process_batch(val_words)

        # Fetches
        fetches_val = [accuracy_valid, recurrent_speller.y_valid]
        feed_dict_val = {
            recurrent_speller.inputs: c_inputs_numerical_val,
            recurrent_speller.input_lengths: c_input_lengths_val,
            recurrent_speller.targets_in: c_targets_in_numerical_val,
            recurrent_speller.targets_out: c_targets_out_numerical_val,
            recurrent_speller.targets_lengths: c_target_lengths_val,
            recurrent_speller.targets_mask: c_targets_mask_val
        }

        # Run validation
        res = tuple(sess.run(fetches=fetches_val, feed_dict=feed_dict_val))
        acc_val, output_val = res
        accs_val += [acc_val]
        print(". Validation accuracy: {:.2%}.".format(acc_val))

        # Plot validation
        ax1.plot(val_batch_nr, accs_val, 'b-')
        ax1.set_ylabel('Validation Accuracy', color="blue")
        ax1.set_xlabel('Batches')
        ax1.set_title('Validation', fontsize=20)
        ax1.grid('on')
        ax1.set_ylim(0, 1)
        ax1.set_xlim(0, n_batches)
        plt.draw()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.pause(1)

save_fig(Path(output_dir, "training_graph"))

print("Storing Recurrent Encoder")
recurrent_speller.save_encoder(sess=sess,
                               file_path=str(Path(output_dir, "checkpoint", "speller_encode.ckpt")))

print("Storing Recurrent Decoder")
recurrent_speller.save_decoder(sess=sess,
                               file_path=str(Path(output_dir, "checkpoint", "speller_decode.ckpt")))


print("Running on test-set! (may be too big to handle)")
# Process validation batch
c_test_words = random.sample(words_test, test_batch_size)
(c_word_sample_test, c_input_lengths_test, c_target_lengths_test, c_inputs_numerical_test,
 c_targets_in_numerical_test,
 c_targets_out_numerical_test, c_targets_mask_test) = recurrent_speller.process_batch(c_test_words)

# Fetches
fetches_test = [accuracy_valid, recurrent_speller.y_valid]
feed_dict_test = {
    recurrent_speller.inputs: c_inputs_numerical_test,
    recurrent_speller.input_lengths: c_input_lengths_test,
    recurrent_speller.targets_in: c_targets_in_numerical_test,
    recurrent_speller.targets_out: c_targets_out_numerical_test,
    recurrent_speller.targets_lengths: c_target_lengths_test,
    recurrent_speller.targets_mask: c_targets_mask_test
}

# Run test
res = tuple(sess.run(fetches=fetches_test, feed_dict=feed_dict_test))
acc_test, output_test = res
print("\tTest accuracy: {:.2%}".format(acc_test))

print("Storing results to file.")
# Store results to file
results = dict(
    validation_accuracies=[float(val) for val in accs_val],
    test_accuracy=float(acc_test),
    cells=cells,
    learning_rates=learning_rates.tolist(),
    training_batch_size=training_batch_size,
    val_interval=val_interval,
    val_batch_size=val_batch_size,
    test_batch_size=test_batch_size,
    n_batches=n_batches,
)
with Path(output_dir, "results.json").open("w") as file:
    json.dump(results, file)

# pickle.dump(results, Path(output_dir, "results.p").open("wb"))

print('Done')

