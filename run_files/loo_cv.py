import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut

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
    p = 0
    loo = LeaveOneOut()
    for train, test in loo.split(unique_programs):
        # Get split indices
        train_idx = program_ids != unique_programs[test]
        test_idx = program_ids == unique_programs[test]

        print('Welcome to program %i' % (p + 1))
        print('Number of training examples %i' % (np.sum(train_idx)))
        print('Number of test examples %i' % (np.sum(test_idx)))

        #
        training_data = tensor_provider.load_data_tensors(train_idx)
        test_data = tensor_provider.load_data_tensors(test_idx)
        m = 0
        for model in model_list:
            # TODO: Yes we have problems with TF-sessions.
            # TODO:   The speller in tensor_provider uses a sessions that should not be reset.
            # Initalizize TF.seesion... and clear previous?
            tfsess = tf.Session()
            model.fit(training_data, tfsess)
            y_pred = model.predict(test_data, tfsess)

            # Evaluate with eval_functions
            e = 0
            for evalf in eval_functions:
                classification_results[p, m, e] = evalf(test_data['labels'], y_pred)
                e += 1

            m += 1

        print("Done with training and evaluation! ---")
        p += 1
    return classification_results


if __name__ == "__main__":
    the_tensor_provider = TensorProvider(tf_graph=tf.Graph(), tf_session=tf.Session(), verbose=True)

    leave_one_program_out_cv(tensor_provider=the_tensor_provider, model_list=[])
