import shutil
from pathlib import Path

import numpy as np
import xarray as xr

from models.model_base import DetektorModel
from project_paths import ProjectPaths
from evaluations.area_roc import ROC, plot_roc
from models.baselines import LogisticRegression, MLP, LogisticRegressionSK, SVMSK, GaussianProcess
from evaluations import Accuracy, F1, TruePositives, TrueNegatives, FalsePositives, FalseNegatives, Samples, \
    AreaUnderROC
from models.recurrent.basic_recurrent import BasicRecurrent
from models.PositiveLearningElkan.pu_learning import PULogisticRegressionSK
from util.learning_rate_utilities import linear_geometric_curve
from util.tensor_provider import TensorProvider
from util.utilities import ensure_folder, save_fig, redirect_stdout_to_file, close_stdout_file
from datetime import datetime


def single_training(tensor_provider, model,
                    test_programs, eval_functions=None, return_predictions=False):
    """
    :param TensorProvider tensor_provider: Class providing all data to models.
    :param DetektorModel model: Model-class to train and test.
    :param list test_programs: List of program IDs used for test-programs
    :param list[Evaluation] eval_functions: List of evaluation functions used to test models.
    :param bool return_predictions: If True, the method stores all model test-predictions and returns them as well.
                                    Can be used to determine whether errors are the same across models.
    :return:
    """
    # Default evaluation score
    if eval_functions is None:
        eval_functions = [Accuracy(), F1(), TruePositives(), TrueNegatives(), FalsePositives(), FalseNegatives(),
                          Samples(), AreaUnderROC(), ROC()]

    # Elements keys
    keys = list(sorted(tensor_provider.accessible_annotated_keys))

    # Get program ids and number of programs
    program_ids = np.array(list(zip(*keys))[0])

    # Initialize array for holding results
    special_results_train = dict()
    evaluation_names = [val.name() for val in eval_functions if val.is_single_value]
    classification_results_train = np.full((1, len(evaluation_names)), np.nan)
    classification_results_train = xr.DataArray(classification_results_train,
                                                name="Training Results",
                                                dims=["Model", "Evaluation"],
                                                coords=dict(Evaluation=evaluation_names,
                                                            Model=[model.name]))
    special_results_test = dict()
    classification_results_test = np.full((1, len(evaluation_names)), np.nan)
    classification_results_test = xr.DataArray(classification_results_test,
                                               name="Test Results",
                                               dims=["Model", "Evaluation"],
                                               coords=dict(Evaluation=evaluation_names,
                                                           Model=[model.name]))

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

    # Convert to keys
    train_idx = [keys[val] for val in train_idx]
    test_idx = [keys[val] for val in test_idx]

    # Make and set BoW-vocabulary
    bow_vocabulary = tensor_provider.extract_programs_vocabulary(train_idx)
    tensor_provider.set_bow_vocabulary(bow_vocabulary)

    # Get truth of train-set
    y_true_train = tensor_provider.load_labels(data_keys_or_idx=train_idx)

    # Get truth of test-set
    y_true = tensor_provider.load_labels(data_keys_or_idx=test_idx)

    # Initialize model
    model.initialize_model(tensor_provider=tensor_provider)

    # Fit model
    model.fit(tensor_provider=tensor_provider,
              train_idx=train_idx,
              verbose=2)

    # Predict on training-data
    print("\tPredicting on training data")
    y_pred_train, y_pred_train_binary = model.predict(tensor_provider=tensor_provider,
                                                      predict_idx=train_idx)
    y_pred_train = np.squeeze(y_pred_train)
    y_pred_train_binary = np.squeeze(y_pred_train_binary)

    train_predictions = y_pred_train

    # Predict on test-data for performance
    print("\tPredicting on test data")
    y_pred, y_pred_binary = model.predict(tensor_provider=tensor_provider,
                                          predict_idx=test_idx)
    y_pred = np.squeeze(y_pred)
    y_pred_binary = np.squeeze(y_pred_binary)

    # Store predictions
    test_predictions = y_pred

    # Evaluate with eval_functions
    print("\tRunning evaluation functions")
    evaluation_nr = 0
    for evalf in eval_functions:
        # Training evaluation
        assert y_pred_train.shape == y_true_train.shape, "y_pred ({}) and y_true ({}) " \
                                                         "do not have same shape".format(y_pred_train.shape,
                                                                                         y_true_train.shape)

        if evalf.is_single_value:
            evaluation_result = evalf(y_true=y_true_train,
                                      y_pred=y_pred_train,
                                      y_pred_binary=y_pred_train_binary)
            classification_results_train[0, evaluation_nr] = evaluation_result
        else:
            special_results_train[(model.name, evalf.name())] = evalf(y_true=y_true_train,
                                                                      y_pred=y_pred_train,
                                                                      y_pred_binary=y_pred_train_binary)

        # Test evaluation
        assert y_pred.shape == y_true.shape, "y_pred ({}) and y_true ({}) " \
                                             "do not have same shape".format(y_pred.shape, y_true.shape)

        if evalf.is_single_value:
            evaluation_result = evalf(y_true=y_true,
                                      y_pred=y_pred,
                                      y_pred_binary=y_pred_binary)
            classification_results_test[0, evaluation_nr] = evaluation_result
            evaluation_nr += 1
        else:
            special_results_test[(model.name, evalf.name())] = evalf(y_true=y_true,
                                                                     y_pred=y_pred,
                                                                     y_pred_binary=y_pred_binary)

    # Save model
    print("\tSaving model")
    model.save_model()

    # Return list
    returns = [classification_results_train, classification_results_test,
               special_results_train, special_results_test, model.summary_to_string()]

    # Additional returns
    if return_predictions:
        returns.extend([train_predictions, test_predictions])

    return tuple(returns)


if __name__ == "__main__":

    # Initialize tensor-provider (data-source)
    the_tensor_provider = TensorProvider(verbose=True)

    # Results path
    results_path = Path(ProjectPaths.results, "single_train")

    # Choose model
    # model = GaussianProcess(
    #     tensor_provider=the_tensor_provider,
    #     use_bow=True,
    #     use_embedsum=True,
    #     verbose=True,
    #     results_path=results_path,
    #     n_jobs=-1
    # )
    n_test_programs = 2
    n_batches = 2000
    learning_rates = linear_geometric_curve(n=n_batches,
                                            starting_value=1e-2,
                                            end_value=1e-8,
                                            geometric_component=3. / 4,
                                            geometric_end=5)
    a_model = BasicRecurrent(
        tensor_provider=the_tensor_provider,
        results_path=results_path,
        n_batches=n_batches,
        recurrent_units=50,
        linear_units=[],
        learning_rate_progression=learning_rates,
        name_formatter="{}_hey"
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

    # Create model-specific path and ensure directory
    results_path = a_model.create_model_path(results_path=results_path)
    shutil.rmtree(str(results_path))
    ensure_folder(results_path)
    redirect_stdout_to_file(Path(results_path, "log.txt"))
    print("Script starting at: {}".format(datetime.now().strftime("%d-%m-%Y %H:%M:%S")))

    # Select test-programs
    unique_programs = np.array(sorted(set(the_tensor_provider.accessible_annotated_program_ids)))
    test_programs = np.random.choice(unique_programs, size=n_test_programs, replace=False)

    # Run training on a single model
    results_train, results_test, \
        s_results_train, s_results_test, \
        model_summary = single_training(
            tensor_provider=the_tensor_provider,
            model=a_model,
            test_programs=test_programs
        )  # type: xr.DataArray

    # Print mean results
    results_train = results_train._to_dataset_split("Model").to_dataframe()
    results_test = results_test._to_dataset_split("Model").to_dataframe()
    with Path(results_path, "results.txt").open("w") as file:
        file.write(model_summary)
        file.write(str(results_train))
        file.write(str(results_test))

    print("\nSingle training Results - TRAINING \n" + "-" * 75)
    print(results_train)
    print("\nSingle training Results - TEST \n" + "-" * 75)
    print(results_test)
    print("\nModel Summary \n" + "-" * 75)
    print(model_summary)

    # Plot ROC if included
    roc_key = (a_model.name, "ROC")
    if roc_key in s_results_train:
        positive_rate, negative_rate = s_results_train[roc_key]
        plot_roc(tp_rate=positive_rate,
                 fp_rate=negative_rate,
                 title="{} ROC Training".format(a_model.name))
        save_fig(Path(results_path, "ROC_Train"))

    if roc_key in s_results_test:
        positive_rate, negative_rate = s_results_test[roc_key]
        plot_roc(tp_rate=positive_rate,
                 fp_rate=negative_rate,
                 title="{} ROC Test".format(a_model.name))
        save_fig(Path(results_path, "ROC_Test"))

    print("Script ended at: {}".format(datetime.now().strftime("%d-%m-%Y %H:%M:%S")))
    close_stdout_file()
