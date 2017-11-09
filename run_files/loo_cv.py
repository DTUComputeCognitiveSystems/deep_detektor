import numpy as np
import tensorflow as tf
from sklearn.model_selection import LeaveOneOut
import xarray as xr

from models.model_base import DetektorModel
from models.baselines import MLP, LogisticRegression
from evaluations import Accuracy, F1, TruePositives, TrueNegatives, FalsePositives, FalseNegatives, Samples
from util.tensor_provider import TensorProvider


def leave_one_program_out_cv(tensor_provider, model_list, eval_functions=None, limit=None, return_predictions=False):
    """
    :param TensorProvider tensor_provider: Class providing all data to models.
    :param list[ClassVar] model_list: List of model-classes for testing.
    :param list[Evaluation] eval_functions: List of evaluation functions used to test models.
    :param bool return_predictions: If True, the method stores all model test-predictions and returns them as well.
                                    Can be used to determine whether errors are the same across models.
    :param int | None limit: Only perform analysis on some programs (for testing)
                             If None - run on all programs.
    :return:
    """
    n_models = len(model_list)

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
    n_programs = len(unique_programs)
    program_names = ["P{:02d}".format(val + 1) for val in range(n_programs)]

    # Dictionary for holding actual predictions (they vary in length which discourages an array)
    test_predictions = dict()

    # Initialize array for holding results
    classification_results = np.full((n_programs, n_models, n_evaluations), np.nan)
    classification_results = xr.DataArray(classification_results,
                                          name="Loo Results",
                                          dims=["Program", "Model", "Evaluation"],
                                          coords=dict(Program=program_names,
                                                      Model=[model_class.name() for model_class in model_list],
                                                      Evaluation=[val.name() for val in eval_functions]))

    # TODO: Is this loop-ordering counter-intuitive? Would it be better to train and measure each model once at a time?

    # Loop over programs
    loo = LeaveOneOut()
    limit = len(unique_programs) if limit is None else limit
    print("\n\nRunning Leave-One-Out Tests.\n" + "-" * 75)
    for program_nr, (train, test) in enumerate(list(loo.split(unique_programs))[:limit]):
        program_name = program_names[program_nr]

        # Get split indices
        train_idx = np.where(program_ids != unique_programs[test])[0]
        test_idx = np.where(program_ids == unique_programs[test])[0]

        print("Program {}, using {} training samples and {} test samples.".format(program_nr + 1,
                                                                                  len(train_idx),
                                                                                  len(test_idx)))

        # Make BoW-vocabulary
        bow_vocabulary = tensor_provider.extract_programs_vocabulary(train_idx)

        # TODO: These should be moved into the models training-facilities, enabling mini-batches
        # Get data-tensors
        training_data = tensor_provider.load_data_tensors(train_idx,
                                                          bow_vocabulary=bow_vocabulary,
                                                          word_embedding=False,
                                                          char_embedding=False,
                                                          pos_tags=False)
        test_data = tensor_provider.load_data_tensors(test_idx,
                                                      bow_vocabulary=bow_vocabulary,
                                                      word_embedding=False,
                                                      char_embedding=False,
                                                      pos_tags=False)

        # Go through models
        for model_nr, model_class in enumerate(model_list):
            model_name = model_class.name()
            print(" " * 2 + "Model: {}".format(model_name))

            # Initialize TF.session... and clear previous?
            tf_sess = tf.Session()

            # Initialize model
            model = model_class()  # type: DetektorModel

            # Fit model
            model.fit(training_data, tf_sess, indentation=4)

            # Predict on test-data for performance
            y_pred = np.squeeze(model.predict(test_data, tf_sess))
            y_true = test_data['labels']

            # Store predictions
            if return_predictions:
                test_predictions.setdefault(model_name, dict())[program_name] = y_pred

            # Evaluate with eval_functions
            for evaluation_nr, evalf in enumerate(eval_functions):
                assert y_pred.shape == y_true.shape, "y_pred ({}) and y_true ({}) " \
                                                     "does not have same shape".format(y_pred.shape, y_true.shape)
                evaluation_results = evalf(y_true, y_pred)
                classification_results[program_nr, model_nr, evaluation_nr] = evaluation_results

        print(" " * 2 + "Done with program {}".format(program_nr + 1))

    if return_predictions:
        return classification_results, test_predictions
    return classification_results


if __name__ == "__main__":
    the_tensor_provider = TensorProvider(verbose=True)

    results = leave_one_program_out_cv(tensor_provider=the_tensor_provider,
                                       model_list=[LogisticRegression, MLP],
                                       limit=None)  # type: xr.DataArray

    mean_results = results.mean("Program")
    mean_results.name = "Mean Loo Results"

    print("\nMean LOO Results\n" + "-" * 75)
    print(mean_results._to_dataset_split("Model").to_dataframe())
