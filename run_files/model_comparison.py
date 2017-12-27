from pathlib import Path
import tensorflow as tf

import numpy as np
import shutil

from models.baselines import LogisticRegression, LogisticRegressionSK, SVMSK
from models.recurrent.basic_recurrent import BasicRecurrent
from project_paths import ProjectPaths
from run_files.single_train import single_training
from util.learning_rate_utilities import linear_geometric_curve
from util.tensor_provider import TensorProvider

# TODO: Make the single-train use the same programs every time (easier comparison)
# TODO: Make single_train loop over a list of test-programs (like loo_cv)
# TODO: Make some of the scripts have the option of initializing the same model multiple times and train multiple times
# TODO:     and then select the best model.

if __name__ == "__main__":

    # Initialize tensor-provider (data-source)
    the_tensor_provider = TensorProvider(verbose=True)

    # Results path
    base_path = Path(ProjectPaths.results, "model_comparison")

    # Settings
    n_test_programs = 2
    n_runs = 4

    # Select test-programs
    unique_programs = np.array(sorted(set(the_tensor_provider.accessible_annotated_program_ids)))
    used_test_programs = np.random.choice(unique_programs, size=n_test_programs, replace=False)
    used_training_programs = np.array(sorted(set(unique_programs).difference(set(used_test_programs))))

    ################
    # Re-used settings

    # BasicRecurrent
    n_batches = 4000
    standard_recurrent_settings = dict(
        tensor_provider=the_tensor_provider,
        results_path=base_path,
        n_batches=n_batches,
        learning_rate_progression=linear_geometric_curve(
            n=n_batches,
            starting_value=1e-4,
            end_value=1e-10,
            geometric_component=3. / 4,
            geometric_end=5
        )
    )

    # LogisticRegressionSK

    # Remove everything there is in base-directory
    if base_path.is_dir():
        shutil.rmtree(str(base_path))

    ################
    # MULTIPLE RUNS

    # Paths for all models
    model_paths = []

    # Header
    print("-" * 75)
    print("-" * 100)
    print("\t\t\t\tModel Comparison")
    print("-" * 100)
    print("-" * 75, end="\n\n")

    # Runs
    for run_nr in range(n_runs):
        print("-" * 75)
        print("Run {}".format(run_nr))
        print("-" * 75, end="\n\n")

        ####
        # List of models

        # Models
        model_list = [
            BasicRecurrent(
                **standard_recurrent_settings,
                recurrent_units=400,
                feedforward_units=[200],
                dropouts=[1],
                recurrent_neuron_type=tf.nn.rnn_cell.GRUCell,
                training_curve_y_limit=1000
            ),
            BasicRecurrent(
                **standard_recurrent_settings,
                recurrent_units=500,
                feedforward_units=[250],
                dropouts=[1],
                recurrent_neuron_type=tf.nn.rnn_cell.GRUCell,
                training_curve_y_limit=1000
            ),
            BasicRecurrent(
                **standard_recurrent_settings,
                recurrent_units=400,
                feedforward_units=[200, 100],
                dropouts=[1, 2],
                recurrent_neuron_type=tf.nn.rnn_cell.GRUCell,
                training_curve_y_limit=1000
            ),
            BasicRecurrent(
                **standard_recurrent_settings,
                recurrent_units=500,
                feedforward_units=[250, 100],
                dropouts=[1, 2],
                recurrent_neuron_type=tf.nn.rnn_cell.GRUCell,
                training_curve_y_limit=1000
            ),
            # With static features
            BasicRecurrent(
                **standard_recurrent_settings,
                use_bow=True,
                recurrent_units=400,
                feedforward_units=[200],
                dropouts=[-1, 1],
                recurrent_neuron_type=tf.nn.rnn_cell.GRUCell,
                training_curve_y_limit=1000
            ),
            BasicRecurrent(
                **standard_recurrent_settings,
                use_bow=True,
                recurrent_units=500,
                feedforward_units=[250],
                dropouts=[-1, 1],
                recurrent_neuron_type=tf.nn.rnn_cell.GRUCell,
                training_curve_y_limit=1000
            ),
            BasicRecurrent(
                **standard_recurrent_settings,
                use_bow=True,
                recurrent_units=400,
                feedforward_units=[200, 100],
                dropouts=[-1, 1, 2],
                recurrent_neuron_type=tf.nn.rnn_cell.GRUCell,
                training_curve_y_limit=1000
            ),
            BasicRecurrent(
                **standard_recurrent_settings,
                use_bow=True,
                recurrent_units=500,
                feedforward_units=[250, 100],
                dropouts=[-1, 1, 2],
                recurrent_neuron_type=tf.nn.rnn_cell.GRUCell,
                training_curve_y_limit=1000
            ),
        ]

        ################

        for model_nr, a_model in enumerate(model_list):
            print("\n\n" + "-"*100)
            print("Model {}: {}\n".format(model_nr, a_model.name))

            # Run training on a single model
            single_training(
                tensor_provider=the_tensor_provider,
                model=a_model,
                test_programs=used_test_programs,
                training_programs=used_training_programs,
                base_path=base_path
            )
            model_paths.append(a_model.create_model_path(base_path))

    print("\nModels created at paths:")
    for path in model_paths:
        print("\t{}".format(path))
