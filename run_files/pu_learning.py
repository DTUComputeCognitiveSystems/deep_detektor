import shutil
import sqlite3
from pathlib import Path

import numpy as np

from collections import defaultdict
import matplotlib.pyplot as plt

from evaluations import Accuracy, F1, AreaUnderROC
from models.baselines import LogisticRegressionSK
from project_paths import ProjectPaths
from util.tensor_provider import TensorProvider
from util.utilities import ensure_folder, redirect_stdout_to_file, save_fig

# Initialize tensor-provider (data-source)
the_tensor_provider = TensorProvider(verbose=True)

###########
# Settings

# Make a model
model = LogisticRegressionSK(
    tensor_provider=the_tensor_provider,
    use_bow=True, use_embedsum=True
)

# How confident should we be to include negatives?
reliable_negative_threshold = 0.99

# Number of iterations for program
n_iterations = 20

# Path for results
results_path = Path(ProjectPaths.results, "pu_learning_{}".format(model.name()))

###########

# Clean and ensure directory
try:
    shutil.rmtree(str(results_path))
except FileNotFoundError:
    pass
ensure_folder(results_path)

# Log file
log_path = Path(results_path, "log.txt")
redirect_stdout_to_file(log_path)

# Dictionary for holding progression of labels
label_progression = defaultdict(lambda: [])
confidence_progression = defaultdict(lambda: [])

# Get all negative and unlabelled data
all_keys = the_tensor_provider.accessible_keys
labels = the_tensor_provider.load_labels(all_keys)

# Get original sentences
sentences = the_tensor_provider.load_original_sentences(all_keys)

# Assume all negative and unlabelled to be negative
labels = [False if val is None else val for val in labels]

# Get negative keys and positive keys
u_keys = [key for idx, key in enumerate(all_keys) if not labels[idx]]
p_keys = [key for idx, key in enumerate(all_keys) if labels[idx]]
n_p = len(p_keys)

# Note initial labels
for key, label in zip(all_keys, labels):
    label_progression[key].append("P" if label else "N")
    confidence_progression[key].append(0 if label else 1)

# Current keys and labels
c_keys = all_keys
c_labels = labels

print("\nRunning Positive-Unlabelled Learning.")
for iteration_nr in range(n_iterations):
    print("\tIteration {}".format(iteration_nr))
    print("\t\tStats:")
    print("\t\t\tTraining with {} samples".format(len(c_keys)))
    print("\t\t\tPositives: {}".format(len([val for val in c_labels if val])))
    print("\t\t\tNegatives: {}".format(len([val for val in c_labels if not val])))

    # Initialise model
    print("\t\tInitializing model")
    model.initialize_model(tensor_provider=the_tensor_provider)
    with Path(results_path, "settings.txt").open("w") as file:
        file.write(model.summary_to_string())

    # Fit model
    print("\t\tFitting model")
    model.fit(tensor_provider=the_tensor_provider,
              train_idx=c_keys,
              y=c_labels)

    # Run on training data
    print("\t\tRunning on training data")
    y_pred, y_binary = model.predict(
        tensor_provider=the_tensor_provider,
        predict_idx=c_keys
    )

    # How did you do?
    print("\t\tTraining performance")
    for evalf in [Accuracy(), F1(), AreaUnderROC()]:
        print("\t\t\t{} : {:.6f}".format(evalf.name(), evalf(c_labels, y_pred, y_binary)))

    # Run on negative data
    print("\t\tRunning on all unlabelled/negative data")
    y_pred, _ = model.predict(
        tensor_provider=the_tensor_provider,
        predict_idx=u_keys
    )

    # Sort negatives
    print("\t\tZipping and sorting keys and their labels")
    y_pred = 1 - np.squeeze(y_pred)
    confidence_and_keys = list(zip(y_pred, u_keys))
    key2confidence = {key: val for val, key in zip(y_pred, u_keys)}
    sorted_confidence_and_keys = list(sorted(confidence_and_keys))

    # Split
    print("\t\tSplitting into negative and unknown")
    reliable_negative_keys = [key for val, key in sorted_confidence_and_keys if val >= reliable_negative_threshold]
    continued_unlabelled_keys = [key for val, key in sorted_confidence_and_keys if val < reliable_negative_threshold]
    print("\t\t\tReliable negatives: {}".format(len(reliable_negative_keys)))
    print("\t\t\tUnlabelled: {}".format(len(continued_unlabelled_keys)))

    # Note new labels labels
    for key in p_keys:
        label_progression[key].append("P")
        confidence_progression[key].append(0)
    for key in reliable_negative_keys:
        label_progression[key].append("N")
        confidence_progression[key].append(key2confidence[key])
    for key in continued_unlabelled_keys:
        label_progression[key].append("-")
        confidence_progression[key].append(key2confidence[key])

    # New data
    c_keys = p_keys + reliable_negative_keys
    c_labels = [True] * len(p_keys) + [False] * len(reliable_negative_keys)

##################
# Make database
database_path = Path(results_path, "pu_learning_label_progression.db")
connection = sqlite3.connect(str(database_path))
cursor = connection.cursor()

##################
# Store label-history as database

# Make table
labels_command = "".join(["label_{} INTEGER NOT NULL,".format(idx) for idx in range(n_iterations + 1)])
command = (
        "CREATE TABLE labels ("
        "program_id INTEGER NOT NULL,"
        "sentence_id INTEGER NOT NULL,"
        "sentence TEXT NOT NULL," +
        labels_command +
        "PRIMARY KEY (program_id, sentence_id)"
        ")"
)
cursor.execute(command)

# Make data for database
data = [
    (key[0], key[1], sentence, *[val for val in label_progression[key]])
    for key, sentence in zip(all_keys, sentences)
]

# Insert information
labels_command = ",".join(["label_{}".format(idx) for idx in range(n_iterations + 1)])
command = "INSERT INTO labels (program_id, sentence_id, sentence, {}) VALUES ({})" \
    .format(labels_command, ",".join(["?" for val in range(4 + n_iterations)]))
cursor.executemany(command, data)
connection.commit()


##################
# Store label-confidence-history as database (identical to above but with confidences)

# Make table
confidences_command = "".join(["confidence_{} REAL NOT NULL,".format(idx) for idx in range(n_iterations + 1)])
command = (
        "CREATE TABLE confidences ("
        "program_id INTEGER NOT NULL,"
        "sentence_id INTEGER NOT NULL,"
        "sentence TEXT NOT NULL," +
        confidences_command +
        "PRIMARY KEY (program_id, sentence_id)"
        ")"
)
cursor.execute(command)

# Make data for database
data = [
    (key[0], key[1], sentence, *[val for val in confidence_progression[key]])
    for key, sentence in zip(all_keys, sentences)
]

# Insert information
confidences_command = ",".join(["confidence_{}".format(idx) for idx in range(n_iterations + 1)])
command = "INSERT INTO confidences (program_id, sentence_id, sentence, {}) VALUES ({})" \
    .format(confidences_command, ",".join(["?" for val in range(4 + n_iterations)]))
cursor.executemany(command, data)
connection.commit()


##################
# Store reliable negatives, positives and unlabelled samples to database

# Make table
command = (
        "CREATE TABLE final_labels ("
        "program_id INTEGER NOT NULL,"
        "sentence_id INTEGER NOT NULL,"
        "sentence TEXT NOT NULL,"
        "label TEXT NOT NULL,"
        "PRIMARY KEY (program_id, sentence_id)"
        ")"
)
cursor.execute(command)

# Make data for database
data = []
for key, sentence in zip(all_keys, sentences):
    if all([val == "N" for val in label_progression[key]]):
        label = "N"
    elif all([val == "P" for val in label_progression[key]]):
        label = "P"
    else:
        label = "U"

    data.append((key[0], key[1], sentence, label))

# Insert information
command = "INSERT INTO final_labels (program_id, sentence_id, sentence, label) VALUES (?, ?, ?, ?)"
cursor.executemany(command, data)
connection.commit()


##################
# Close database
cursor.close()
connection.close()

##################
# Plot history

plt.close("all")
plt.figure()

# Label matrix
label_matrix = np.int32(np.array([
    [val == "N" for val in label_progression[key]]
    for key in all_keys
]))

# History-sum
negatives_history = label_matrix.sum(0)
unlabelled_history = label_matrix.shape[0] - negatives_history
positives_history = np.ones(label_matrix.shape[1]) * n_p

# X
x = np.array(list(range(label_matrix.shape[1])))

# Plot curves
plt.plot(x, negatives_history, label="Negatives")
plt.plot(x, unlabelled_history, label="Unlabelled")
plt.plot(x, positives_history, label="Positives [{}]".format(n_p))

# Labels and title
plt.title("Labels history.")
plt.xlabel("Iteration number")
plt.ylabel("# Labels")
plt.legend()

# Save
save_fig(Path(results_path, "aggregated_label_history"), only_pdf=True)

# Do log
ax = plt.gca()
ax.set_yscale("log", nonposy='clip')

# Save log-version
save_fig(Path(results_path, "aggregated_label_history_log"), only_pdf=True)

