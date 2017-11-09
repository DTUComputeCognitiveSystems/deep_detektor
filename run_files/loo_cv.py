import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut

from models.model_base import DetektorModel
from models.baselines import MLP, LogisticRegression
from util.tensor_provider import TensorProvider


def leave_one_program_out_cv(tensor_provider, model_list, eval_functions=None, limit=None):
    """
    :param TensorProvider tensor_provider:
    :param list[ClassVar] model_list:
    :param list[Callable] eval_functions:
    :return:
    """

    # Default evaluation score
    if eval_functions is None:
        eval_functions = [accuracy_score]

    # Elements keys
    keys = tensor_provider.keys

    # Get program ids and number of programs
    program_ids = np.array(list(zip(*keys))[0])
    unique_programs = np.array(sorted(set(program_ids)))
    n_programs = len(unique_programs)

    # Initialize array for holding results
    classification_results = np.full((n_programs, len(model_list), len(eval_functions)), np.nan)

    # TODO: Is this loop-ordering counter-intuitive? Would it be better to train and measure each model once at a time?

    # Loop over programs
    loo = LeaveOneOut()
    limit = len(unique_programs) if limit is None else limit
    print("\n\nRunning Leave-One-Out Tests.\n" + "-" * 75)
    for program_nr, (train, test) in enumerate(list(loo.split(unique_programs))[:limit]):
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
            # Initialize TF.session... and clear previous?
            tf_sess = tf.Session()

            # Initialize model
            model = model_class()  # type: DetektorModel
            print(" " * 2 + "Model: {}".format(model.name))

            # Fit model
            model.fit(training_data, tf_sess, indentation=4)

            # Predict on test-data for performance
            y_pred = model.predict(test_data, tf_sess)

            # Evaluate with eval_functions
            for evaluation_nr, evalf in enumerate(eval_functions):
                classification_results[program_nr, model_nr, evaluation_nr] = evalf(test_data['labels'], y_pred)

        print(" " * 2 + "Done with program {}".format(program_nr + 1))

    return classification_results


if __name__ == "__main__":
    the_tensor_provider = TensorProvider(verbose=True)

    results = leave_one_program_out_cv(tensor_provider=the_tensor_provider,
                                       model_list=[LogisticRegression, MLP],
                                       limit=None)

    print("\nResults\n" + "-" * 75)
    print(results)
