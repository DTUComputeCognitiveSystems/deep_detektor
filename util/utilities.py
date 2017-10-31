# Import
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf

def leave_one_program_out_cv(data, model_list, eval_functions=[accuracy_score]):
    # TODO: Change model_list to class_list and initialize objects in model-loop
    # Get program ids and number of programs
    program_ids = data['data'][:, 2]
    unique_programs = np.unique(program_ids)
    NUM_PROGRAMS = len(unique_programs)
    loo = LeaveOneOut()
    classification_results = np.empty((NUM_PROGRAMS, len(model_list), len(eval_functions)))
    classification_results.fill(np.nan)

    # Loop over programs
    p = 0
    for train, test in loo.split(unique_programs):
        train_idx = program_ids != unique_programs[test]
        test_idx = program_ids == unique_programs[test]

        print('Welcome to program %i' % (p + 1))
        print('Number of training examples %i' % (np.sum(train_idx)))
        print('Number of test examples %i' % (np.sum(test_idx)))

        training_data, test_data = data_to_tensors(data, train_idx, test_idx)
        m = 0
        for model in model_list:
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


def data_to_tensors(data, train_indices=None, test_indices=None):
    # TODO: embedding input (string 'bow', 'word2vec', 'glove', ... )
    # Performs neccesary feature extraction and test/training split
    # Returns data transformed in multiple ways tensors:
    #    char: Char-based
    #    pos: Part-of-Speech tagging
    #    word2vec: Word2vec (or someother subspace..)
    #    bow: Bag-Of-Words
    # Furthermore returns for each sample the binary vector y (labels)

    data_train = dict()
    data_test = dict()

    # Extract relevant data from table
    X = data['data'][:, 4]  # sentences
    y = data['data'][:, 6]  # claim indices
    N = len(X)

    # If no test/train split is specified return everything in training
    if train_indices is None and test_indices is None:
        train_indices = np.ones(N, dtype=bool)

    # Pr. sample label-vector
    y = np.asarray([y[i] is not None for i in range(N)])
    data_train['labels'] = y[train_indices]
    if train_indices is not None and test_indices is not None:
        data_test['labels'] = y[test_indices]

    # Char
    # ...

    # Pos
    # ...

    # Word2Vec
    # ...

    # Bag-Of-Words
    # TODO: Should not retrain bag-of-words for every test/train split?
    # TODO: Bag-of-words should be in SQL database?
    vectorizer = CountVectorizer()
    X_bow = vectorizer.fit_transform(X)
    data_train['bow'] = X_bow[train_indices, :]
    if train_indices is not None and test_indices is not None:
        data_test['bow'] = X_bow[test_indices, :]

    return data_train, data_test