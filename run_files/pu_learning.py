from pathlib import Path
import numpy as np

from evaluations import Accuracy, F1, AreaUnderROC
from models.baselines import LogisticRegressionSK
from models.recurrent.basic_recurrent import BasicRecurrent
from project_paths import ProjectPaths
from util.tensor_provider import TensorProvider

n_test = None
reliable_negative_threshold = 0.99

# Initialize tensor-provider (data-source)
the_tensor_provider = TensorProvider(verbose=True)

# Get all negative and unlabelled data
keys = the_tensor_provider.accessible_keys
labels = the_tensor_provider.load_labels(keys)

# Assume all negative and unlabelled to be negative
labels = [False if val is None else val for val in labels]

# Make a model
model = LogisticRegressionSK(
     tensor_provider=the_tensor_provider,
)

# Initialise model
model.initialize_model(tensor_provider=the_tensor_provider)

# Fit model
model.fit(tensor_provider=the_tensor_provider,
          train_idx=keys,
          verbose=2,
          y=labels)

# Run on all data
y_pred, y_binary = model.predict(
    tensor_provider=the_tensor_provider,
    predict_idx=keys
)

# How did you do?
print("Training performance")
for evalf in [Accuracy(), F1(), AreaUnderROC()]:
    print(evalf.name(), ":", evalf(labels, y_pred, y_binary))

# Get negative keys
negative_keys = [key for idx, key in enumerate(keys) if not labels[idx]]
if n_test is not None:
    negative_keys = negative_keys[:n_test]

# Run on negative data
y_pred, y_binary = model.predict(
    tensor_provider=the_tensor_provider,
    predict_idx=negative_keys
)
y_pred = 1-np.squeeze(y_pred)

# Sort negatives
confidence_and_keys = list(zip(y_pred, negative_keys))
sorted_confidence_and_keys = list(sorted(confidence_and_keys))

# Split
reliable_negative_keys = [key for val, key in sorted_confidence_and_keys if val >= reliable_negative_threshold]
continued_unlabelled_keys = [key for val, key in sorted_confidence_and_keys if val < reliable_negative_threshold]
