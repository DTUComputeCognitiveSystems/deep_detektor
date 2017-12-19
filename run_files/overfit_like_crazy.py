import xarray as xr
from pathlib import Path
from evaluations import Accuracy, F1, TruePositives, TrueNegatives, FalsePositives, FalseNegatives, Samples, \
    Precision, AreaUnderROC
import numpy as np
from util.learning_rate_utilities import linear_geometric_curve
from models.recurrent.basic_recurrent import BasicRecurrent
from models.baselines import LogisticRegression, MLP, SVMSK
from util.tensor_provider import TensorProvider
from util.sql_utilities import rows2sql_table
from util.utilities import ensure_folder, redirect_stdout_to_file
from project_paths import ProjectPaths
import tensorflow as tf

overfit_like_crazy_directory = Path(ProjectPaths.results, "overfit_like_crazy")

###################################
# Settings

# Test-train parameters
n_test_programs = 2
n_train_programs = 1

# Initialize tensor-provider (data-source)
the_tensor_provider = TensorProvider(verbose=True)

# Initialize model
# REC_HIDDEN_UNITS = 200
# FC_HIDDEN_UNITS = 400
# ITERS=2000
# BATCH_SIZE=100
# recmodel = BasicRecurrent(the_tensor_provider, units=[REC_HIDDEN_UNITS, FC_HIDDEN_UNITS],
#                          optimizer=tf.train.AdamOptimizer, word_embedding=True,
#                           pos_tags=True, char_embedding=False)

# model = LogisticRegression(the_tensor_provider,  use_bow=True, use_embedsum=False,
#                 learning_rate=0.001, training_epochs=100, verbose=False)
# model = MLP(
#     tensor_provider=the_tensor_provider,
#     hidden_units=10,
#     learning_rate=0.01,
#     training_epochs=4000,
#     verbose=False,
#     use_bow=True,
#     use_embedsum=True,
#     class_weights=np.array([1.0, 1.0])
# )
n_batches = 5000
learning_rates = linear_geometric_curve(n=n_batches,
                                        starting_value=1e-2,
                                        end_value=1e-8,
                                        geometric_component=3. / 4,
                                        geometric_end=5)
model = BasicRecurrent(
    tensor_provider=the_tensor_provider,
    recurrent_units=100,
    linear_units=[100, 50],
    word_embedding=True,
    pos_tags=True,
    char_embedding=True,
    n_batches=n_batches,
    batch_size=64,
    display_step=1,
    results_path=overfit_like_crazy_directory,
    learning_rate_progression=learning_rates
)

# Evaluation functions
eval_functions = [Samples(), TruePositives(), TrueNegatives(), FalsePositives(), FalseNegatives(),
                  Accuracy(), Precision(), F1(), AreaUnderROC()]
n_evaluations = len(eval_functions)

###################################
# Work

# Save prints to file
log_path = Path(overfit_like_crazy_directory, "log.txt")
redirect_stdout_to_file(log_path)

# Print model summary
print(model.autosummary_str())

# Make data-array for training results
classification_results_training = np.full((1, n_evaluations), np.nan)
classification_results_training = xr.DataArray(classification_results_training,
                                               name="Training Results",
                                               dims=["Model", "Evaluation"],
                                               coords=dict(Evaluation=[val.name() for val in eval_functions],
                                                           Model=[model.name()]))

# Make data-array for test results
classification_results_test = np.full((1, n_evaluations), np.nan)
classification_results_test = xr.DataArray(classification_results_test,
                                           name="Test Results",
                                           dims=["Model", "Evaluation"],
                                           coords=dict(Evaluation=[val.name() for val in eval_functions],
                                                       Model=[model.name()]))

# Elements keys
keys = list(sorted(the_tensor_provider.accessible_annotated_keys))

# Get program ids and number of programs
program_ids = np.array(list(zip(*keys))[0])
unique_programs = set(program_ids)
assert n_test_programs + n_train_programs <= len(unique_programs)

# Select test-programs and training-programs
test_programs = np.random.choice(list(unique_programs),
                                 size=n_test_programs,
                                 replace=False)
training_programs = np.random.choice(list(unique_programs.difference(set(test_programs))),
                                     size=n_train_programs,
                                     replace=False)

# Get test-indices
test_idx = np.sum([program_ids == val for val in test_programs], axis=0)
test_idx = np.where(test_idx > 0.5)[0]

# Get test-indices
train_idx = np.sum([program_ids == val for val in training_programs], axis=0)
train_idx = np.where(train_idx > 0.5)[0]

# Check that nothing overlaps (sanity check)
assert not set(test_idx).intersection(set(train_idx))

# Convert to keys
train_idx = [keys[val] for val in train_idx]
test_idx = [keys[val] for val in test_idx]

# Report
print("{} available programs with {} samples.".format(len(unique_programs), len(program_ids)))
print("Training programs {} with {} samples.".format(training_programs, len(train_idx)))
print("Test programs {} with {} samples.".format(test_programs, len(test_idx)))

# Get truth of test-set and train
y_true_train = the_tensor_provider.load_labels(data_keys_or_idx=train_idx)
y_true_test = the_tensor_provider.load_labels(data_keys_or_idx=test_idx)

###################################
# Training

print("Training model.")

# Initialize mode
model.initialize_model(tensor_provider=the_tensor_provider)

# Fit model
model.fit(
    tensor_provider=the_tensor_provider,
    train_idx=train_idx,
    verbose=2
)

###################################
# Evaluation

# Predict on train-data
print("Running on training data.")
y_pred_train, y_pred_train_binary = model.predict(tensor_provider=the_tensor_provider, predict_idx=train_idx)
y_pred_train = np.squeeze(y_pred_train)
y_pred_train_binary = np.squeeze(y_pred_train_binary)

# Predict on test-data for performance
print("Running on test data.")
y_pred_test, y_pred_test_binary = model.predict(tensor_provider=the_tensor_provider, predict_idx=test_idx)
y_pred_test = np.squeeze(y_pred_test)
y_pred_test_binary = np.squeeze(y_pred_test_binary)

# Evaluate with eval_functions
print("Running evaluation measures,")
for evaluation_nr, evalf in enumerate(eval_functions):
    # Training data evaluation
    assert y_pred_train.shape == y_true_train.shape, "Training: y_pred ({}) and y_true ({}) " \
                                                     "do not have same shape".format(y_pred_train.shape,
                                                                                     y_true_train.shape)

    evaluation_results_train = evalf(y_true_train, y_pred_train, y_pred_train_binary)
    classification_results_training[0, evaluation_nr] = evaluation_results_train

    # Hold-out test data evaluation
    assert y_pred_test.shape == y_true_test.shape, "Test: y_pred ({}) and y_true ({}) " \
                                                   "do not have same shape".format(y_pred_test.shape,
                                                                                   y_true_test.shape)

    evaluation_results_test = evalf(y_true_test, y_pred_test, y_pred_test_binary)
    classification_results_test[0, evaluation_nr] = evaluation_results_test

# Print mean results
print("\nSingle training Results -- TRAINING --\n" + "-" * 75)
print(classification_results_training._to_dataset_split("Model").to_dataframe())

print("\nSingle training Results -- TEST --\n" + "-" * 75)
print(classification_results_test._to_dataset_split("Model").to_dataframe())

print("\nModel Summary --\n" + "-" * 75)
print(model.summary_to_string())

###################################
# Storage

# Make path for database
database_path = Path(overfit_like_crazy_directory, "results.db")
ensure_folder(database_path)

# Data for results-database
headers = [
    "name",
    "n_train_programs",
    "n_test_programs",
    *["{}_train".format(val.name()) for val in eval_functions],
    *["{}_test".format(val.name()) for val in eval_functions],
    "model_str"
]
results_data = [
    model.name(),
    n_train_programs,
    n_test_programs,
    *classification_results_training.data.tolist()[0],
    *classification_results_test.data.tolist()[0],
    model.autosummary_str()
]

# Append results
assert len(headers) == len(results_data)
rows2sql_table(
    data=[results_data],
    database_path=database_path,
    table_name="results",
    column_headers=headers,
    append=True
)
