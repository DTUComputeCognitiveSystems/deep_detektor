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

# TODO: Make the single-train use the same programs everytime (easier comparison)
# TODO: Make single_traing loop over a list of test-programs (like loov_cv)
# TODO: Make some of the scripts have the option of initializing the same model multiple times and train multiple times
# TODO:     and then select the best model.

if __name__ == "__main__":

    # Initialize tensor-provider (data-source)
    the_tensor_provider = TensorProvider(verbose=True)

    # Results path
    base_path = Path(ProjectPaths.results, "model_comparison")

    # Settings
    n_test_programs = 2
    n_runs = 2

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
    # List of models

    # Models
    model_list = [
        BasicRecurrent(
            **standard_recurrent_settings,
            recurrent_units=400,
            linear_units=[200],
            name_formatter="{}_400_200_drop1_a",
            dropouts=[1],
            recurrent_neuron_type=tf.nn.rnn_cell.GRUCell,
            training_curve_y_limit=1000
        ),
        BasicRecurrent(
            **standard_recurrent_settings,
            recurrent_units=500,
            linear_units=[250],
            name_formatter="{}_500_250_drop1_a",
            dropouts=[1],
            recurrent_neuron_type=tf.nn.rnn_cell.GRUCell,
            training_curve_y_limit=1000
        ),
        BasicRecurrent(
            **standard_recurrent_settings,
            recurrent_units=400,
            linear_units=[200, 100],
            name_formatter="{}_400_200_100_drop1_a",
            dropouts=[1],
            recurrent_neuron_type=tf.nn.rnn_cell.GRUCell,
            training_curve_y_limit=1000
        ),
        BasicRecurrent(
            **standard_recurrent_settings,
            recurrent_units=500,
            linear_units=[250, 100],
            name_formatter="{}_500_250_100_drop1_a",
            dropouts=[1],
            recurrent_neuron_type=tf.nn.rnn_cell.GRUCell,
            training_curve_y_limit=1000
        ),
        # AGAIN
        BasicRecurrent(
            **standard_recurrent_settings,
            recurrent_units=400,
            linear_units=[200],
            name_formatter="{}_400_200_drop1_b",
            dropouts=[1],
            recurrent_neuron_type=tf.nn.rnn_cell.GRUCell,
            training_curve_y_limit=1000
        ),
        BasicRecurrent(
            **standard_recurrent_settings,
            recurrent_units=500,
            linear_units=[250],
            name_formatter="{}_500_250_drop1_b",
            dropouts=[1],
            recurrent_neuron_type=tf.nn.rnn_cell.GRUCell,
            training_curve_y_limit=1000
        ),
        BasicRecurrent(
            **standard_recurrent_settings,
            recurrent_units=400,
            linear_units=[200, 100],
            name_formatter="{}_400_200_100_drop1_b",
            dropouts=[1],
            recurrent_neuron_type=tf.nn.rnn_cell.GRUCell,
            training_curve_y_limit=1000
        ),
        BasicRecurrent(
            **standard_recurrent_settings,
            recurrent_units=500,
            linear_units=[250, 100],
            name_formatter="{}_500_250_100_drop1_b",
            dropouts=[1],
            recurrent_neuron_type=tf.nn.rnn_cell.GRUCell,
            training_curve_y_limit=1000
        ),
        # BasicRecurrent(
        #     **standard_recurrent_settings,
        #     recurrent_units=15,
        #     name_formatter="{}_15_rnn",
        # ),
        # BasicRecurrent(
        #     **standard_recurrent_settings,
        #     recurrent_units=20,
        #     name_formatter="{}_20_rnn",
        # ),
        # BasicRecurrent(
        #     **standard_recurrent_settings,
        #     recurrent_units=25,
        #     name_formatter="{}_25_rnn",
        # ),
        # BasicRecurrent(
        #     **standard_recurrent_settings,
        #     recurrent_units=30,
        #     name_formatter="{}_30_rnn",
        # ),
        # # Dropout:
        # BasicRecurrent(
        #     **standard_recurrent_settings,
        #     recurrent_units=20,
        #     name_formatter="{}_20_rnn_drop",
        #     dropouts=[0]
        # ),
        # BasicRecurrent(
        #     **standard_recurrent_settings,
        #     recurrent_units=25,
        #     name_formatter="{}_25_rnn_drop",
        #     dropouts=[0]
        # ),
        # BasicRecurrent(
        #     **standard_recurrent_settings,
        #     recurrent_units=30,
        #     name_formatter="{}_30_rnn_drop",
        #     dropouts=[0]
        # ),
        # # Linear units
        # BasicRecurrent(
        #     **standard_recurrent_settings,
        #     recurrent_units=15,
        #     name_formatter="{}_15_rnn_10_lin",
        #     linear_units=[10],
        # ),
        # BasicRecurrent(
        #     **standard_recurrent_settings,
        #     recurrent_units=20,
        #     name_formatter="{}_20_rnn_10_lin",
        #     linear_units=[10],
        # ),
        # BasicRecurrent(
        #     **standard_recurrent_settings,
        #     recurrent_units=25,
        #     name_formatter="{}_25_rnn_10_lin",
        #     linear_units=[10],
        # ),
        # # Linear units and dropout
        # BasicRecurrent(
        #     **standard_recurrent_settings,
        #     recurrent_units=15,
        #     name_formatter="{}_15_rnn_10_lin_drop",
        #     linear_units=[10],
        #     dropouts=[0]
        # ),
        # BasicRecurrent(
        #     **standard_recurrent_settings,
        #     recurrent_units=20,
        #     name_formatter="{}_20_rnn_10_lin_drop",
        #     linear_units=[10],
        #     dropouts=[0]
        # ),
        # BasicRecurrent(
        #     **standard_recurrent_settings,
        #     recurrent_units=25,
        #     name_formatter="{}_25_rnn_10_lin_drop",
        #     linear_units=[10],
        #     dropouts=[0]
        # ),
        # # Baselines
        # LogisticRegressionSK(
        #     tensor_provider=the_tensor_provider,
        #     use_bow=True,
        #     use_embedsum=True,
        #     tol=1e-7,
        #     max_iter=500,
        #     name_formatter="{}_bow_and_embedsum"
        # ),
        # LogisticRegressionSK(
        #     tensor_provider=the_tensor_provider,
        #     use_bow=True,
        #     use_embedsum=False,
        #     tol=1e-7,
        #     max_iter=500,
        #     name_formatter="{}_bow"
        # ),
        # LogisticRegressionSK(
        #     tensor_provider=the_tensor_provider,
        #     use_bow=False,
        #     use_embedsum=True,
        #     tol=1e-7,
        #     max_iter=500,
        #     name_formatter="{}_embedsum"
        # ),
        # SVMSK(
        #     tensor_provider=the_tensor_provider,
        #     use_bow=True,
        #     use_embedsum=True,
        #     name_formatter="{}_bow_and_embedsum"
        # ),
    ]

    ################

    print("-"*50 + "\n" + "-" * 100 + "\n\t\t\t\tModel Comparison\n" + "-"*100 + "\n" + "-"*50)

    model_paths = []
    for model_nr, a_model in enumerate(model_list):
        print("\n\n" + "-"*100)
        print("Model {}: {}\n".format(model_nr, a_model.name))

        if n_runs == 1:
            # Run training on a single model
            single_training(
                tensor_provider=the_tensor_provider,
                model=a_model,
                test_programs=used_test_programs,
                training_programs=used_training_programs,
                base_path=base_path
            )
            model_paths.append(a_model.create_model_path(base_path))

        else:
            name_path_formatter = a_model.name + "_{}"

            for run_nr in range(n_runs):
                a_model.set_name(name_path_formatter.format(chr(ord("a") + run_nr)))

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
