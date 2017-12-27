import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from models.PositiveLearningElkan.pu_learning import PULogisticRegressionSK
from models.baselines import LogisticRegressionSK
from models.recurrent.basic_recurrent import BasicRecurrent
from project_paths import ProjectPaths
from run_files.single_train import single_training
from util.learning_rate_utilities import linear_geometric_curve
from util.tensor_provider import TensorProvider

if __name__ == "__main__":
    # Initialize tensor-provider (data-source)
    the_tensor_provider = TensorProvider(verbose=True)

    # Results path
    used_base_path = Path(ProjectPaths.results, "single_train")

    # Settings
    test_ratio = 0.11

    # Models
    n_batches = 2000
    learning_rates = linear_geometric_curve(n=n_batches,
                                            starting_value=5e-4,
                                            end_value=1e-10,
                                            geometric_component=3. / 4,
                                            geometric_end=5)
    a_model = BasicRecurrent(
        tensor_provider=the_tensor_provider,
        results_path=used_base_path,
        use_bow=False,
        n_batches=n_batches,
        batch_size=64,
        learning_rate_progression=learning_rates,
        recurrent_units=400,
        feedforward_units=[200],
        dropouts=[1],
        recurrent_neuron_type=tf.nn.rnn_cell.GRUCell,
        training_curve_y_limit=1000
    )
    # a_model = LogisticRegression(
    #     tensor_provider=the_tensor_provider,
    # )
    # a_model = MLP(
    #     tensor_provider=the_tensor_provider,
    # )
    # a_model = SVMSK(
    #     tensor_provider=the_tensor_provider,
    #     verbose=True
    # )
    # a_model = LogisticRegressionSK(
    #      tensor_provider=the_tensor_provider,
    # )
    # a_model = PULogisticRegressionSK(
    #     tensor_provider=the_tensor_provider,
    # )

    # Select random sentences for training and test
    keys = the_tensor_provider.accessible_annotated_keys
    random.shuffle(keys)
    loc_split = int(len(keys) * test_ratio)
    training_keys = keys[loc_split:]
    test_keys = keys[:loc_split]

    # Run training on a single model
    single_training(
        tensor_provider=the_tensor_provider,
        model=a_model,
        test_split=test_keys,
        training_split=training_keys,
        base_path=used_base_path,
        split_is_keys=True
    )
