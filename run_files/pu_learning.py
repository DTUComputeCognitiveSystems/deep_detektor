import shutil
import sqlite3
from pathlib import Path

import numpy as np

from collections import defaultdict
from evaluations import Accuracy, F1, AreaUnderROC
from models.baselines import LogisticRegressionSK
from project_paths import ProjectPaths
from util.tensor_provider import TensorProvider
from util.utilities import ensure_folder


###########
# Settings

# How confident should we be to include negatives?
reliable_negative_threshold = 0.99

# Number of iterations for program
n_iterations = 3

# Path for results
results_path = Path(ProjectPaths.results, "pu_learning")

###########

# Clean and ensure directory
try:
    shutil.rmtree(str(results_path))
except FileNotFoundError:
    pass
ensure_folder(results_path)

# Dictionary for holding progression of labels
label_progression = defaultdict(lambda: [])

# Initialize tensor-provider (data-source)
the_tensor_provider = TensorProvider(verbose=True)

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

# Note initial labels
for key, label in zip(all_keys, labels):
    label_progression[key].append("P" if label else "N")

# Current keys and labels
c_keys = all_keys
c_labels = labels

print("\nRunning Positive-Unlabelled Learning.")
for iteration_nr in range(n_iterations):
    print("\tIteration {}".format(iteration_nr))

    # Make a model
    print("\t\tMaking model")
    model = LogisticRegressionSK(
        tensor_provider=the_tensor_provider,
    )

    # Initialise model
    model.initialize_model(tensor_provider=the_tensor_provider)

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
        print("\t\t\t{} : {}".format(evalf.name(), evalf(c_labels, y_pred, y_binary)))

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
    sorted_confidence_and_keys = list(sorted(confidence_and_keys))

    # Split
    print("\t\tSplitting into negative and unknown")
    reliable_negative_keys = [key for val, key in sorted_confidence_and_keys if val >= reliable_negative_threshold]
    continued_unlabelled_keys = [key for val, key in sorted_confidence_and_keys if val < reliable_negative_threshold]

    # Note new labels labels
    for key in p_keys:
        label_progression[key].append("P")
    for key in reliable_negative_keys:
        label_progression[key].append("N")
    for key in continued_unlabelled_keys:
        label_progression[key].append("-")

    # New data
    c_keys = p_keys + reliable_negative_keys
    c_labels = [True] * len(p_keys) + [False] * len(reliable_negative_keys)

##################
# Store result as database

# Make database
database_path = Path(results_path, "pu_learning_label_progression.db")
connection = sqlite3.connect(str(database_path))
cursor = connection.cursor()

# Make table
labels_command = "".join(["label_{} INTEGER NOT NULL,".format(idx) for idx in range(n_iterations+1)])
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
labels_command = ",".join(["label_{}".format(idx) for idx in range(n_iterations+1)])
command = "INSERT INTO labels (program_id, sentence_id, sentence, {}) VALUES ({})" \
        .format(labels_command, ",".join(["?" for val in range(4+n_iterations)]))
cursor.executemany(command, data)
connection.commit()

# Close database
cursor.close()
connection.close()
