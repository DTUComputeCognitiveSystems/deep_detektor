from pathlib import Path

import numpy as np
import shutil

from models.baselines import LogisticRegression
from project_paths import ProjectPaths
from run_files.single_train import single_training
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

    # Remove everything there is in base-directory
    if base_path.is_dir():
        shutil.rmtree(str(base_path))

    # Models
    model_list = [
        LogisticRegression(
            the_tensor_provider,
            use_bow=True,
            use_embedsum=False,
            training_epochs=10,
            learning_rate=0.1,
            batch_size=400,
            batch_strategy="weighted_sampling",
            name_formatter="{}_only_bow"
        ),
        LogisticRegression(
            the_tensor_provider,
            use_bow=False,
            use_embedsum=True,
            training_epochs=10,
            learning_rate=0.001,
            batch_size=400,
            batch_strategy="weighted_sampling",
            name_formatter="{}_only_embedsum"
        ),
        LogisticRegression(
            the_tensor_provider,
            use_bow=False,
            use_embedsum=True,
            training_epochs=10,
            learning_rate=0.001,
            batch_size=400,
            batch_strategy="weighted_sampling",
            name_formatter="{}_bow_and_embedsum"
        )
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
