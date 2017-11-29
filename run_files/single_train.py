from pathlib import Path

import numpy as np
import xarray as xr

from project_paths import ProjectPaths
from evaluations.area_roc import ROC, plot_roc
from models.baselines import LogisticRegression, MLP
from evaluations import Accuracy, F1, TruePositives, TrueNegatives, FalsePositives, FalseNegatives, Samples, \
    AreaUnderROC
from models.recurrent.basic_recurrent import BasicRecurrent
from util.tensor_provider import TensorProvider
from util.utilities import ensure_folder, save_fig


def single_training(tensor_provider, model_class,
                    n_test_programs=1, eval_functions=None, return_predictions=False,
                    path=None):
    """
    :param TensorProvider tensor_provider: Class providing all data to models.
    :param ClassVar model_class: Model-class to train and test.
    :param int n_test_programs: Number of test-programs to use.
    :param list[Evaluation] eval_functions: List of evaluation functions used to test models.
    :param bool return_predictions: If True, the method stores all model test-predictions and returns them as well.
                                    Can be used to determine whether errors are the same across models.
    :param Path path: Path for storing results.
    :return:
    """
    # Default evaluation score
    if eval_functions is None:
        eval_functions = [Accuracy(), F1(), TruePositives(), TrueNegatives(), FalsePositives(), FalseNegatives(),
                          Samples(), AreaUnderROC(), ROC()]

    # Elements keys
    keys = tensor_provider.keys

    # Get program ids and number of programs
    program_ids = np.array(list(zip(*keys))[0])
    unique_programs = np.array(sorted(set(program_ids)))

    # Initialize array for holding results
    special_results_train = dict()
    evaluation_names = [val.name() for val in eval_functions if val.is_single_value]
    classification_results_train = np.full((1, len(evaluation_names)), np.nan)
    classification_results_train = xr.DataArray(classification_results_train,
                                          name="Training Results",
                                          dims=["Model", "Evaluation"],
                                          coords=dict(Evaluation=evaluation_names,
                                                      Model=[model_class.name()]))
    special_results_test = dict()
    classification_results_test = np.full((1, len(evaluation_names)), np.nan)
    classification_results_test = xr.DataArray(classification_results_test,
                                          name="Test Results",
                                          dims=["Model", "Evaluation"],
                                          coords=dict(Evaluation=evaluation_names,
                                                      Model=[model_class.name()]))

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

    # Make and set BoW-vocabulary
    bow_vocabulary = tensor_provider.extract_programs_vocabulary(train_idx)
    tensor_provider.set_bow_vocabulary(bow_vocabulary)

    # Get truth of train-set
    y_true_train = tensor_provider.load_labels(data_keys_or_idx=train_idx)

    # Get truth of test-set
    y_true = tensor_provider.load_labels(data_keys_or_idx=test_idx)

    # Initialize model
    model = model_class(tensor_provider=tensor_provider)

    # Fit model
    model.fit(tensor_provider=tensor_provider,
              train_idx=train_idx,
              verbose=2,
              results_path=path)

    # Predict on training-data
    y_pred_train, y_pred_train_binary = model.predict(tensor_provider=tensor_provider,
                                          predict_idx=train_idx)
    y_pred_train = np.squeeze(y_pred_train)
    y_pred_train_binary = np.squeeze(y_pred_train_binary)

    train_predictions = y_pred_train

    # Predict on test-data for performance
    y_pred, y_pred_binary = model.predict(tensor_provider=tensor_provider,
                                          predict_idx=test_idx)
    y_pred = np.squeeze(y_pred)
    y_pred_binary = np.squeeze(y_pred_binary)

    # Store predictions
    test_predictions = y_pred

    # Evaluate with eval_functions
    evaluation_nr = 0
    for evalf in eval_functions:
        # Training evaluation
        assert y_pred_train.shape == y_true_train.shape, "y_pred ({}) and y_true ({}) " \
                                             "do not have same shape".format(y_pred_train.shape, y_true_train.shape)

        if evalf.is_single_value:
            evaluation_result = evalf(y_true=y_true_train,
                                      y_pred=y_pred_train,
                                      y_pred_binary=y_pred_train_binary)
            classification_results_train[0, evaluation_nr] = evaluation_result
        else:
            special_results_train[(model_class.name(), evalf.name())] = evalf(y_true=y_true_train,
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
            special_results_test[(model_class.name(), evalf.name())] = evalf(y_true=y_true,
                                                                        y_pred=y_pred,
                                                                        y_pred_binary=y_pred_binary)

    if return_predictions:
        return classification_results_train, classification_results_test, \
               special_results_train, special_results_test, model.summary_to_string(), \
               train_predictions, test_predictions
    return classification_results_train, classification_results_test, \
           special_results_train, special_results_test, model.summary_to_string()


if __name__ == "__main__":
    # Initialize tensor-provider (data-source)
    the_tensor_provider = TensorProvider(verbose=True)

    # Choose model
    #model = LogisticRegression
    model = MLP

    # Results path
    results_path = Path(ProjectPaths.results, "single_train", model.name())
    ensure_folder(results_path)

    # Run training on a single model
    results_train, results_test, \
    s_results_train, s_results_test, \
    model_summary = single_training(tensor_provider=the_tensor_provider,
                                         model_class=model,
                                         path=results_path)  # type: xr.DataArray

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
    roc_key = (model.name(), "ROC")
    if roc_key in s_results_train:
        positive_rate, negative_rate = s_results_train[roc_key]
        plot_roc(tp_rate=positive_rate,
                 fp_rate=negative_rate,
                 title="{} ROC Training".format(model.name()))
        save_fig(Path(results_path, "ROC_Train"))

    if roc_key in s_results_test:
        positive_rate, negative_rate = s_results_test[roc_key]
        plot_roc(tp_rate=positive_rate,
                 fp_rate=negative_rate,
                 title="{} ROC Test".format(model.name()))
        save_fig(Path(results_path, "ROC_Test"))