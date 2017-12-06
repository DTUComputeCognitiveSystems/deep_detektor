from pathlib import Path

import numpy as np
import xarray as xr

from project_paths import ProjectPaths
from evaluations.area_roc import ROC, plot_roc
from models.baselines import LogisticRegression, MLP
from evaluations import Accuracy, F1, TruePositives, TrueNegatives, FalsePositives, FalseNegatives, Samples, \
    AreaUnderROC
from util.tensor_provider import TensorProvider
from util.utilities import ensure_folder, save_fig


def model_comparison(tensor_provider, model_list,
                    test_idx, train_idx, eval_functions=None, return_predictions=False,
                    path=None):
    """
    :param TensorProvider tensor_provider: Class providing all data to models.
    :param list[DetektorModel] model_list: List of initialized Detektor Models
    :param list[int] test_programs: List of test-programs to use (indices)
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

    # Initialize array for holding results
    special_results_train = dict()
    evaluation_names = [val.name() for val in eval_functions if val.is_single_value]
    model_names = [mod.name() for mod in model_list]
    classification_results_train = np.full((len(model_list), len(evaluation_names)), np.nan)
    classification_results_train = xr.DataArray(classification_results_train,
                                          name="Training Results",
                                          dims=["Model", "Evaluation"],
                                          coords=dict(Evaluation=evaluation_names,
                                                      Model=model_names))
    special_results_test = dict()
    classification_results_test = np.full((len(model_list), len(evaluation_names)), np.nan)
    classification_results_test = xr.DataArray(classification_results_test,
                                          name="Test Results",
                                          dims=["Model", "Evaluation"],
                                          coords=dict(Evaluation=evaluation_names,
                                                      Model=model_names))


    # Get truth of train-set
    y_true_train = tensor_provider.load_labels(data_keys_or_idx=train_idx)

    # Get truth of test-set
    y_true = tensor_provider.load_labels(data_keys_or_idx=test_idx)

    #Loop over models...
    for m, model in enumerate(model_list):
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
                classification_results_train[m, evaluation_nr] = evaluation_result
            else:
                special_results_train[(m, evalf.name())] = evalf(y_true=y_true_train,
                                                                            y_pred=y_pred_train,
                                                                            y_pred_binary=y_pred_train_binary)

            # Test evaluation
            assert y_pred.shape == y_true.shape, "y_pred ({}) and y_true ({}) " \
                                                 "do not have same shape".format(y_pred.shape, y_true.shape)

            if evalf.is_single_value:
                evaluation_result = evalf(y_true=y_true,
                                          y_pred=y_pred,
                                          y_pred_binary=y_pred_binary)
                classification_results_test[m, evaluation_nr] = evaluation_result
                evaluation_nr += 1
            else:
                special_results_test[(m, evalf.name())] = evalf(y_true=y_true,
                                                                            y_pred=y_pred,
                                                                            y_pred_binary=y_pred_binary)

    with Path(path, "results.txt").open("w") as file:
        for m in range(len(model_list)):
            rtrain = classification_results_train[m,].to_dataframe()
            rtest = classification_results_test[m,].to_dataframe()
            file.write(model_list[m].summary_to_string())
            file.write(str(rtrain))
            file.write(str(rtest))
            file.write("\n \n")
            file.write("-" * 75 + "\n")
            file.write("\n \n")

    if return_predictions:
        return classification_results_train, classification_results_test, \
               special_results_train, special_results_test, \
               train_predictions, test_predictions
    return classification_results_train, classification_results_test, \
           special_results_train, special_results_test


if __name__ == "__main__":
    # Test programs
    test_programs_idx = [0]

    # Initialize tensor-provider (data-source)
    the_tensor_provider = TensorProvider(verbose=True)

    # Elements keys
    keys = the_tensor_provider.keys

    # Get program ids and number of programs
    program_ids = np.array(list(zip(*keys))[0])
    unique_programs = np.array(sorted(set(program_ids)))

    # Select test-programs
    test_programs = unique_programs[test_programs_idx]

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
    bow_vocabulary = the_tensor_provider.extract_programs_vocabulary(train_idx)
    the_tensor_provider.set_bow_vocabulary(bow_vocabulary)

    # Choose models
    model_list = [LogisticRegression(the_tensor_provider, use_bow=True, use_embedsum=False, training_epochs=100, learning_rate=0.1, batch_size=100)]
    #model_list = [LogisticRegression(the_tensor_provider, use_bow=True, use_embedsum=False, training_epochs=100, learning_rate=0.1),
    #              LogisticRegression(the_tensor_provider, use_bow=False, use_embedsum=True, training_epochs=100, learning_rate=0.1),
    #              MLP(the_tensor_provider, use_bow=True, use_embedsum=True, training_epochs=1000, hidden_units=100, learning_rate=0.1)]

    # Results path
    results_path = Path(ProjectPaths.results, "model_comparison")
    ensure_folder(results_path)

    # Run training on a single model
    results_train, results_test, \
    s_results_train, s_results_test = model_comparison(tensor_provider=the_tensor_provider,
                                         model_list=model_list,
                                         test_idx=test_idx,
                                         train_idx=train_idx,
                                         path=results_path)

    # TODO: Add ROC plot for each method tested
    # # Plot ROC if included
    # roc_key = (model.name(), "ROC")
    # if roc_key in s_results_train:
    #     positive_rate, negative_rate = s_results_train[roc_key]
    #     plot_roc(tp_rate=positive_rate,
    #              fp_rate=negative_rate,
    #              title="{} ROC Training".format(model.name()))
    #     save_fig(Path(results_path, "ROC_Train"))
    #
    # if roc_key in s_results_test:
    #     positive_rate, negative_rate = s_results_test[roc_key]
    #     plot_roc(tp_rate=positive_rate,
    #              fp_rate=negative_rate,
    #              title="{} ROC Test".format(model.name()))
    #     save_fig(Path(results_path, "ROC_Test"))