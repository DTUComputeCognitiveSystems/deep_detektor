import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut

from models.baselines import MyMLP, MyLogisticRegression
from util.tensor_provider import TensorProvider


def leave_one_program_out_cv(tensor_provider, model_list, eval_functions=None):
    """
    :param TensorProvider tensor_provider:
    :param model_list:
    :param eval_functions:
    :return:
    """
    # TODO: Change model_list to class_list and initialize objects in model-loop

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

    # Loop over programs
    loo = LeaveOneOut()
    for program_nr, (train, test) in enumerate(loo.split(unique_programs)):
        # Get split indices
        train_idx = np.where(program_ids != unique_programs[test])[0]
        test_idx = np.where(program_ids == unique_programs[test])[0]

        print('\tProgram {}, using {} training samples and {} test samples.'.format(program_nr + 1,
                                                                                    len(train_idx),
                                                                                    len(test_idx)))

        # Make BoW-vocabulary
        bow_vocabulary = tensor_provider.extract_programs_vocabulary(train_idx)

        # TODO: Currently only BoW is returned, as the remaining are not needed in model
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
            model = model_class()

            # Fit model
            model.fit(training_data, tf_sess)

            # Predict on test-data for performance
            y_pred = model.predict(test_data, tf_sess)

            # Evaluate with eval_functions
            for evaluation_nr, evalf in enumerate(eval_functions):
                classification_results[program_nr, model_nr, evaluation_nr] = evalf(test_data['labels'], y_pred)

        print("Done with training and evaluation! ---")

    return classification_results


if __name__ == "__main__":
    the_tensor_provider = TensorProvider(verbose=True)

    results = leave_one_program_out_cv(tensor_provider=the_tensor_provider,
                                       model_list=[MyLogisticRegression, MyMLP])

    print("\nResults\n" + "-" * 75)
    print(results)
