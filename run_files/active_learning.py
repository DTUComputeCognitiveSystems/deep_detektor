import sqlite3
from pathlib import Path

import numpy as np

from models.baselines import LogisticRegressionSK
from project_paths import ProjectPaths
from util.sql_utilities import rows2sql_table
from util.tensor_provider import TensorProvider
from util.utilities import ensure_folder

# Initialize tensor-provider (data-source)
the_tensor_provider = TensorProvider(verbose=True)
print("")

###########
# Settings

# Path for pu-negatives results
inputs_path = Path(ProjectPaths.results, "pu_learning_LogisticRegressionSKLEARN")

# Negative labels
# None: get labels that have been classified as negative in every single iteration of PU (really fucking certain)
# float: threshold the final confidence of the pu-model with this value (higher means more certain of negativity)
negative_label_scheme = None

###########
# Get split from PU-learning

# Log on to database
database_path = Path(inputs_path, "pu_learning_label_progression.db")
connection = sqlite3.connect(str(database_path))
cursor = connection.cursor()

print("Getting labels using scheme: {}".format(negative_label_scheme))

# Get negatives and positives
if negative_label_scheme is None:

    # Get final labels
    command = (
        "SELECT program_id, sentence_id, label FROM final_labels"
    )
    key_and_labels = cursor.execute(command).fetchall()

    # Split into categories
    positives = [(program_id, sentence_id)
                 for program_id, sentence_id, label in key_and_labels
                 if label == "P"]
    negatives = [(program_id, sentence_id)
                 for program_id, sentence_id, label in key_and_labels
                 if label == "N"]
    unlabelled = [(program_id, sentence_id)
                  for program_id, sentence_id, label in key_and_labels
                  if label == "U"]

else:
    # Get positives
    command = (
        "SELECT program_id, sentence_id FROM final_labels WHERE label = 'P'"
    )
    positives = cursor.execute(command).fetchall()
    positives_set = set(positives)

    # Get column headers from confidences table
    cursor.execute('SELECT * FROM confidences')
    column_names = [description[0] for description in cursor.description]

    # Get final confidence number
    final_confidence_number = max([int(val[11:]) for val in column_names if val.startswith("confidence")])
    final_confidence_header = "confidence_{}".format(final_confidence_number)

    # Get keys and confidences
    command = (
        "SELECT program_id, sentence_id, {} FROM confidences".format(final_confidence_header)
    )
    key_and_labels = cursor.execute(command).fetchall()

    # Filter out positives
    key_and_labels = [val for val in key_and_labels
                      if (val[0], val[1]) not in positives_set]

    # Split into negatives and unlabelled
    negatives = [(val[0], val[1]) for val in key_and_labels if val[2] >= negative_label_scheme]
    unlabelled = [(val[0], val[1]) for val in key_and_labels if val[2] < negative_label_scheme]

# Note sizes of data-splits
n_positives = len(positives)
n_negatives = len(negatives)
n_unlabelled = len(unlabelled)
print("\tPositives: {}".format(n_positives))
print("\tNegatives: {}".format(n_negatives))
print("\tUnlabelled: {}".format(n_unlabelled))
print("\tTotal: {}".format(n_positives + n_negatives + n_unlabelled))

# Close up database
cursor.close()
connection.close()

###########
# Training

print("\nCreating and initializing model")

# Train some model
model = LogisticRegressionSK(
    tensor_provider=the_tensor_provider,
    use_bow=True,
    use_embedsum=True,
    verbose=True,
    tol=1e-7
)

# Initialize model
model.initialize_model(tensor_provider=the_tensor_provider)

# Make training indices and training labels
train_idx = negatives + positives
train_y = np.array([-1] * n_negatives + [1] * n_positives)
assert len(train_idx) == len(train_y)

# Fit model
print("Training model")
model.fit(
    tensor_provider=the_tensor_provider,
    train_idx=train_idx,
    y=train_y
)
print("")

# Run model on unlabelled data
print("Running trained model on unlabeled data")
predictions, binary_predictions = model.predict(tensor_provider=the_tensor_provider, predict_idx=unlabelled)
predictions = predictions.tolist()
binary_predictions = [int(val) for val in binary_predictions]

# Get unlabelled sentences
unlabelled_sentences = the_tensor_provider.load_original_sentences(unlabelled)

# Make directory for results
results_path = Path(ProjectPaths.results, "active_learning_{}".format(model.name()))
ensure_folder(results_path)

# Make data for sql-database
sql_data = [*zip(*unlabelled), predictions, binary_predictions, unlabelled_sentences]
assert all([len(val) == len(sql_data[0]) for val in sql_data])

# Make database
database_path = Path(results_path, "results.db")
rows2sql_table(
    data=sql_data,
    database_path=database_path,
    table_name="predictions",
    column_headers=["program_id", "sentence_id", "predictions", "binary_predictions", "sentence"],
    primary_key=[0, 1],
    data_is_columns=True
)
