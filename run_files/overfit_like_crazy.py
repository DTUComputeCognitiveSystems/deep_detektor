## Overfit to the training data to see if we can learn anything
#exec("../project_paths.py")
import xarray as xr
from evaluations import Accuracy, F1, TruePositives, TrueNegatives, FalsePositives, FalseNegatives, Samples
import numpy as np
from models.recurrent.basic_recurrent import BasicRecurrent
from models.baselines import LogisticRegression, MLP
from util.tensor_provider import TensorProvider
import tensorflow as tf

# Test-train parameters
n_test_programs = 2

# Initialize tensor-provider (data-source)
the_tensor_provider = TensorProvider(verbose=True)

# Initialize model
#REC_HIDDEN_UNITS = 200
#FC_HIDDEN_UNITS = 400
#ITERS=2000
#BATCH_SIZE=100
#recmodel = BasicRecurrent(the_tensor_provider, units=[REC_HIDDEN_UNITS, FC_HIDDEN_UNITS],
#                          optimizer=tf.train.AdamOptimizer, word_embedding=True,
#                           pos_tags=True, char_embedding=False)

#model = LogisticRegression(the_tensor_provider,  use_bow=True, use_embedsum=False,
#                 learning_rate=0.001, training_epochs=100, verbose=False)
model = MLP(the_tensor_provider, hidden_units=10, learning_rate=0.01,
                 training_epochs=50, verbose=False, use_bow=True, use_embedsum=True,
                 class_weights=np.array([1.0, 100.0]))

# Evaluation functions
eval_functions = [Accuracy(), F1(), TruePositives(), TrueNegatives(), FalsePositives(), FalseNegatives(),
                  Samples()]
n_evaluations = len(eval_functions)

classification_results_training = np.full((1, n_evaluations), np.nan)
classification_results_training = xr.DataArray(classification_results_training,
                                          name="Training Results",
                                          dims=["Model", "Evaluation"],
                                          coords=dict(Evaluation=[val.name() for val in eval_functions],
                                                      Model=[model.name()]))

classification_results_test = np.full((1, n_evaluations), np.nan)
classification_results_test = xr.DataArray(classification_results_test,
                                          name="Test Results",
                                          dims=["Model", "Evaluation"],
                                          coords=dict(Evaluation=[val.name() for val in eval_functions],
                                                      Model=[model.name()]))

# Elements keys
keys = the_tensor_provider.keys

# Get program ids and number of programs
program_ids = np.array(list(zip(*keys))[0])
unique_programs = np.array(sorted(set(program_ids)))

# Select test-programs
test_programs = np.random.choice(unique_programs, size=n_test_programs, replace=False)

# Get split indices
test_idx = 0
for test_program in test_programs:
    test_idx = test_idx + (program_ids == test_program)
train_idx = np.where(test_idx < 0.5)[0]
test_idx = np.where(test_idx > 0.5)[0]

# Report
print("Test programs {}, using {} training samples and {} test samples.".format(test_programs + 1,
                                                                           len(train_idx),
                                                                           len(test_idx)))

# Get truth of test-set and train
y_true_train = the_tensor_provider.load_labels(data_keys_or_idx=train_idx)
y_true_test = the_tensor_provider.load_labels(data_keys_or_idx=test_idx)


# Fit model
model.fit(tensor_provider=the_tensor_provider,
          train_idx=train_idx,
          verbose=2)


# Predict on train-data
y_pred_train, y_pred_train_binary = model.predict(tensor_provider=the_tensor_provider,
                       predict_idx=train_idx)
y_pred_train = np.squeeze(y_pred_train)
y_pred_train_binary = np.squeeze(y_pred_train_binary)

# Predict on test-data for performance
y_pred_test, y_pred_test_binary = model.predict(tensor_provider=the_tensor_provider,
                       predict_idx=test_idx)
y_pred_test = np.squeeze(y_pred_test)
y_pred_test_binary = np.squeeze(y_pred_test_binary)

# Evaluate with eval_functions
for evaluation_nr, evalf in enumerate(eval_functions):
    # Training data evaluation
    assert y_pred_train.shape == y_true_train.shape, "Training: y_pred ({}) and y_true ({}) " \
                                                   "do not have same shape".format(y_pred_train.shape, y_true_train.shape)

    evaluation_results_train = evalf(y_true_train, y_pred_train, y_pred_train_binary)
    classification_results_training[0, evaluation_nr] = evaluation_results_train

    # Hold-out test data evaluation
    assert y_pred_test.shape == y_true_test.shape, "Test: y_pred ({}) and y_true ({}) " \
                                         "do not have same shape".format(y_pred_test.shape, y_true_test.shape)

    evaluation_results_test = evalf(y_true_test, y_pred_test, y_pred_test_binary)
    classification_results_test[0, evaluation_nr] = evaluation_results_test



# Print mean results
print("\nSingle training Results -- TRAINING --\n" + "-" * 75)
print(classification_results_training._to_dataset_split("Model").to_dataframe())

print("\nSingle training Results -- TEST --\n" + "-" * 75)
print(classification_results_test._to_dataset_split("Model").to_dataframe())

print("\nModel Summary --\n" + "-" * 75)
print(model.summary_to_string())