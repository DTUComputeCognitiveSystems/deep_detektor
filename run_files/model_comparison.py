from pathlib import Path

import numpy as np
import shutil

from models.baselines import LogisticRegression, LogisticRegressionSK, SVMSK
from models.recurrent.basic_recurrent import BasicRecurrent
from project_paths import ProjectPaths
from run_files.single_train import single_training
from util.learning_rate_utilities import linear_geometric_curve
from util.tensor_provider import TensorProvider

if __name__ == "__main__":

    # Initialize tensor-provider (data-source)
    the_tensor_provider = TensorProvider(verbose=True)

    # Results path
    base_path = Path(ProjectPaths.results, "model_comparison")

    # Settings
    n_test_programs = 2

    # Select test-programs
    unique_programs = np.array(sorted(set(the_tensor_provider.accessible_annotated_program_ids)))
    used_test_programs = np.random.choice(unique_programs, size=n_test_programs, replace=False)
    used_training_programs = np.array(sorted(set(unique_programs).difference(set(used_test_programs))))

    ################
    # Re-used settings

    # BasicRecurrent
    standard_recurrent_settings = dict(
        tensor_provider=the_tensor_provider,
        results_path=base_path,
        n_batches=4000,
        learning_rate_progression=linear_geometric_curve(
            n=4000,
            starting_value=1e-2,
            end_value=1e-8,
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
            recurrent_units=15,
            name_formatter="{}_15_rnn",
        ),
        BasicRecurrent(
            **standard_recurrent_settings,
            recurrent_units=20,
            name_formatter="{}_20_rnn",
        ),
        BasicRecurrent(
            **standard_recurrent_settings,
            recurrent_units=25,
            name_formatter="{}_25_rnn",
        ),
        BasicRecurrent(
            **standard_recurrent_settings,
            recurrent_units=30,
            name_formatter="{}_30_rnn",
        ),
        # Dropout:
        BasicRecurrent(
            **standard_recurrent_settings,
            recurrent_units=20,
            name_formatter="{}_20_rnn_drop",
            dropouts=[0]
        ),
        BasicRecurrent(
            **standard_recurrent_settings,
            recurrent_units=25,
            name_formatter="{}_25_rnn_drop",
            dropouts=[0]
        ),
        BasicRecurrent(
            **standard_recurrent_settings,
            recurrent_units=30,
            name_formatter="{}_30_rnn_drop",
            dropouts=[0]
        ),
        # Linear units
        BasicRecurrent(
            **standard_recurrent_settings,
            recurrent_units=15,
            name_formatter="{}_15_rnn_10_lin",
            linear_units=[10],
        ),
        BasicRecurrent(
            **standard_recurrent_settings,
            recurrent_units=20,
            name_formatter="{}_20_rnn_10_lin",
            linear_units=[10],
        ),
        BasicRecurrent(
            **standard_recurrent_settings,
            recurrent_units=25,
            name_formatter="{}_25_rnn_10_lin",
            linear_units=[10],
        ),
        # Linear units and dropout
        BasicRecurrent(
            **standard_recurrent_settings,
            recurrent_units=15,
            name_formatter="{}_15_rnn_10_lin_drop",
            linear_units=[10],
            dropouts=[0]
        ),
        BasicRecurrent(
            **standard_recurrent_settings,
            recurrent_units=20,
            name_formatter="{}_20_rnn_10_lin_drop",
            linear_units=[10],
            dropouts=[0]
        ),
        BasicRecurrent(
            **standard_recurrent_settings,
            recurrent_units=25,
            name_formatter="{}_25_rnn_10_lin_drop",
            linear_units=[10],
            dropouts=[0]
        ),
        # Baselines
        LogisticRegressionSK(
            tensor_provider=the_tensor_provider,
            use_bow=True,
            use_embedsum=True,
            tol=1e-7,
            max_iter=500,
            name_formatter="{}_bow_and_embedsum"
        ),
        LogisticRegressionSK(
            tensor_provider=the_tensor_provider,
            use_bow=True,
            use_embedsum=False,
            tol=1e-7,
            max_iter=500,
            name_formatter="{}_bow"
        ),
        LogisticRegressionSK(
            tensor_provider=the_tensor_provider,
            use_bow=False,
            use_embedsum=True,
            tol=1e-7,
            max_iter=500,
            name_formatter="{}_embedsum"
        ),
        SVMSK(
            tensor_provider=the_tensor_provider,
            use_bow=True,
            use_embedsum=True,
            name_formatter="{}_bow_and_embedsum"
        ),
    ]

    ################

    model_paths = []
    for a_model in model_list:
        # Run training on a single model
        single_training(
            tensor_provider=the_tensor_provider,
            model=a_model,
            test_programs=used_test_programs,
            training_programs=used_training_programs,
            base_path=base_path
        )
        model_paths.append(a_model.create_model_path(base_path))
