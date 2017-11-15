import numpy as np
import xarray as xr

from models.baselines import LogisticRegression, MLP
from evaluations import Accuracy, F1, TruePositives, TrueNegatives, FalsePositives, FalseNegatives, Samples
from models.recurrent.basic_recurrent import BasicRecurrent
from util.tensor_provider import TensorProvider


def single_training(tensor_provider, model_class, n_test_programs=1, eval_functions=None, return_predictions=False):
    """
    :param TensorProvider tensor_provider: Class providing all data to models.
    :param ClassVar model_class: Model-class to train and test.
    :param int n_test_programs: Number of test-programs to use.
    :param list[Evaluation] eval_functions: List of evaluation functions used to test models.
    :param bool return_predictions: If True, the method stores all model test-predictions and returns them as well.
                                    Can be used to determine whether errors are the same across models.
    :return:
    """
    # Default evaluation score
    if eval_functions is None:
        eval_functions = [Accuracy(), F1(), TruePositives(), TrueNegatives(), FalsePositives(), FalseNegatives(),
                          Samples()]
    n_evaluations = len(eval_functions)

    # Elements keys
    keys = tensor_provider.keys

    # Get program ids and number of programs
    program_ids = np.array(list(zip(*keys))[0])
    unique_programs = np.array(sorted(set(program_ids)))

    # Initialize array for holding results
    classification_results = np.full((1, n_evaluations), np.nan)
    classification_results = xr.DataArray(classification_results,
                                          name="Training Results",
                                          dims=["Model", "Evaluation"],
                                          coords=dict(Evaluation=[val.name() for val in eval_functions],
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

    # Get truth of test-set
    y_true = tensor_provider.load_labels(data_keys_or_idx=test_idx)

    # Initialize model
    model = model_class(tensor_provider=tensor_provider)

    # Fit model
    model.fit(tensor_provider=tensor_provider,
              train_idx=train_idx,
              verbose=2)

    # Predict on test-data for performance
    y_pred = model.predict(tensor_provider=tensor_provider,
                           predict_idx=test_idx)
    y_pred = np.squeeze(y_pred)

    # Store predictions
    test_predictions = y_pred

    # Evaluate with eval_functions
    for evaluation_nr, evalf in enumerate(eval_functions):
        assert y_pred.shape == y_true.shape, "y_pred ({}) and y_true ({}) " \
                                             "do not have same shape".format(y_pred.shape, y_true.shape)
        evaluation_results = evalf(y_true, y_pred)
        classification_results[0, evaluation_nr] = evaluation_results

    if return_predictions:
        return classification_results, test_predictions
    return classification_results


if __name__ == "__main__":
    # Initialize tensor-provider (data-source)
    the_tensor_provider = TensorProvider(verbose=True)

    # Run training on a single model
    results = single_training(tensor_provider=the_tensor_provider,
                              model_class=BasicRecurrent)  # type: xr.DataArray

    # Print mean results
    print("\nSingle training Results\n" + "-" * 75)
    print(results._to_dataset_split("Model").to_dataframe())
