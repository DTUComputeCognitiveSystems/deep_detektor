from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from sklearn.model_selection import LeaveOneOut

from models.model_base import DetektorModel
from project_paths import ProjectPaths
from evaluations import Accuracy, F1, TruePositives, TrueNegatives, FalsePositives, FalseNegatives, Samples, \
    AreaUnderROC
from evaluations.area_roc import plot_roc, ROC, plot_multiple_rocs, mean_rocs
from models.baselines import MLP, LogisticRegression, LogisticRegressionSK
from models.recurrent.basic_recurrent import BasicRecurrent
from util.tensor_provider import TensorProvider
from util.utilities import ensure_folder, save_fig


def leave_one_program_out_cv(tensor_provider, model_list, path,
                             eval_functions=None, limit=None, return_predictions=False,
                             save_ranked_sentences=True, save_full_predictions=True,
                             save_model_weights=True):
    """
    :param TensorProvider tensor_provider: Class providing all data to models.
    :param list[DetektorModel] model_list: List of model-classes for testing.
    :param list[Evaluation] eval_functions: List of evaluation functions used to test models.
    :param bool return_predictions: If True, the method stores all model test-predictions and returns them as well.
                                    Can be used to determine whether errors are the same across models.
    :param int | None limit: Only perform analysis on some programs (for testing)
                             If None - run on all programs.
    :param Path path: Path for storing results
    :return:
    """
    ensure_folder(path)

    # TODO: Consider also looping over loss-functions: classic ones and weighed ones
    n_models = len(model_list)

    # Default evaluation score
    if eval_functions is None:
        eval_functions = [Accuracy(), F1(), TruePositives(), TrueNegatives(), FalsePositives(), FalseNegatives(),
                          Samples(), AreaUnderROC(), ROC()]

    # Elements keys
    keys = list(sorted(tensor_provider.accessible_annotated_keys))

    # Get program ids and number of programs
    program_ids = np.array(list(zip(*keys))[0])
    unique_programs = np.array(sorted(set(program_ids)))
    n_programs = len(unique_programs)
    program_names = ["P{:02d}".format(val + 1) for val in range(n_programs)]

    # Dictionary for holding actual predictions (they vary in length which discourages an array)
    test_predictions = dict()

    # Initialize array for holding results
    special_results = dict()
    evaluation_names = [val.name() for val in eval_functions if val.is_single_value]
    classification_results = np.full((n_programs, n_models, len(evaluation_names)), np.nan)
    classification_results = xr.DataArray(classification_results,
                                          name="Loo Results",
                                          dims=["Program", "Model", "Evaluation"],
                                          coords=dict(Program=program_names,
                                                      Model=[model.name for model in model_list],
                                                      Evaluation=evaluation_names))

    # Initialize file for storing ranked sentences
    if save_ranked_sentences:
        rank_file = Path(path, "ranked_sentences.txt").open("w")

    # Loop over programs
    loo = LeaveOneOut()
    limit = len(unique_programs) if limit is None else limit
    print("\n\nRunning Leave-One-Out Tests.\n" + "-" * 75)
    for program_nr, (train, test) in enumerate(list(loo.split(unique_programs))[:limit]):
        program_name = program_names[program_nr]

        # Get split indices
        train_idx = np.where(program_ids != unique_programs[test])[0]
        test_idx = np.where(program_ids == unique_programs[test])[0]

        # Convert to keys
        train_idx = [keys[val] for val in train_idx]
        test_idx = [keys[val] for val in test_idx]

        # Report
        print("Program {}, using {} training samples and {} test samples.".format(program_nr + 1,
                                                                                  len(train_idx),
                                                                                  len(test_idx)))

        # Make and set BoW-vocabulary
        bow_vocabulary = tensor_provider.extract_programs_vocabulary(train_idx)
        tensor_provider.set_bow_vocabulary(bow_vocabulary)

        # Get truth of test-set
        y_true = tensor_provider.load_labels(data_keys_or_idx=test_idx)

        # Go through models
        for model_nr, model in enumerate(model_list):
            model_name = model.name

            # Initialize model
            model.initialize_model(tensor_provider=tensor_provider)

            # Fit model
            model.fit(tensor_provider=tensor_provider,
                      train_idx=train_idx,
                      verbose=2)

            # Predict on test-data for performance
            y_pred, y_pred_binary = model.predict(tensor_provider=tensor_provider,
                                                  predict_idx=test_idx)
            y_pred = np.squeeze(y_pred)
            y_pred_binary = np.squeeze(y_pred_binary)

            # Store predictions
            if return_predictions:
                test_predictions.setdefault(model_name, dict())[program_name] = y_pred

            # Save the best ranked senteces (in terms of claim)
            if save_ranked_sentences:
                rank_file.write("Test program: %s \n" %program_names[program_nr])
                rank_file.write(model.summary_to_string())
                ranked_sentences, rank_score, rank_indices \
                    = tensor_provider.get_ranked_predictions(y_pred, test_idx)
                rank_file.write("Sentence, Proability of claim, Truth \n")
                ranked_labels = tensor_provider.load_labels(rank_indices)
                for r in range(len(ranked_sentences)):
                    rank_file.write("%s , %.5f, %i \n"%(ranked_sentences[r], rank_score[r], ranked_labels[r]) )
                rank_file.write("\n")

            # Save predictions on full test set
            if save_full_predictions:
                with Path(path, "%s_predictions.txt"%program_names[program_nr]).open("w") as file:
                    all_sentences = tensor_provider.load_original_sentences(test_idx)
                    for r in range(len(all_sentences)):
                        file.write("%i;%.5f;%s\n"%(y_true[r], y_pred[r], all_sentences[r]))

            # Save model weights in case of logistic regression
            if save_model_weights and model_name=="LogisticRegressionSKLEARN":
                #TODO: Save most important weights in classification
                print(' ')


            # Evaluate with eval_functions
            evaluation_nr = 0
            for evalf in eval_functions:
                assert y_pred.shape == y_true.shape, "y_pred ({}) and y_true ({}) " \
                                                     "do not have same shape".format(y_pred.shape, y_true.shape)

                if evalf.is_single_value:
                    evaluation_result = evalf(y_true=y_true,
                                              y_pred=y_pred,
                                              y_pred_binary=y_pred_binary)
                    classification_results[program_nr, model_nr, evaluation_nr] = evaluation_result
                    evaluation_nr += 1
                else:
                    special_results[(model.name, evalf.name(), program_nr)] = evalf(y_true=y_true,
                                                                                             y_pred=y_pred,
                                                                                             y_pred_binary=y_pred_binary)
    ###
    # Plot ROC curves if wanted

    # Go through models
    models_mean_rocs = []
    for model in model_list:
        rocs = []
        labels = []

        # Go through programs
        for program_nr in range(len(unique_programs)):
            key = (model.name, "ROC", program_nr)
            if key in special_results:
                rocs.append(special_results[key])
                labels.append("Program {}".format(program_nr))

        # Plot ROCs for each program for this model
        plot_multiple_rocs(rocs=rocs, labels=labels, center_line=False)
        mean = mean_rocs(rocs)
        models_mean_rocs.append(mean)
        plot_roc(*mean, title=model.name, label="Mean",
                 color="black", linestyle="--")
        plt.legend()

        # Store figure
        file_name = "ROC_{}".format(model.name)
        save_fig(Path(path, file_name))
        plt.close()

    # Plot mean-ROCs for models
    names = [model.name for model in model_list]
    plot_multiple_rocs(rocs=models_mean_rocs, labels=names, center_line=True,
                       title="Models Mean-ROC")
    plt.legend()
    save_fig(Path(path, "Models_ROC"))
    plt.close()

    if save_ranked_sentences:
        rank_file.close()

    if return_predictions:
        return classification_results, special_results, test_predictions
    return classification_results, special_results


if __name__ == "__main__":
    # Initialize tensor-provider (data-source)
    the_tensor_provider = TensorProvider(verbose=True)

    # Choose number of programs to run though (None for all)
    program_limit = None

    # Choose models
    models = [
        LogisticRegressionSK(tensor_provider=the_tensor_provider)
        #MLP(tensor_provider=the_tensor_provider)
    ]

    # Run LOO-program
    loo_path = Path(ProjectPaths.results, "LOO_CV")
    results, s_results = leave_one_program_out_cv(tensor_provider=the_tensor_provider,
                                                  model_list=models,
                                                  limit=program_limit,
                                                  path=loo_path)  # type: xr.DataArray

    # Get mean-results over programs
    mean_results = results.mean("Program")
    mean_results.name = "Mean Loo Results"
    mean_results = mean_results._to_dataset_split("Model").to_dataframe()

    # Print mean results
    print("\nMean LOO Results\n" + "-" * 75)
    with Path(loo_path, "mean_results.txt").open("w") as file:
        file.write(str(mean_results))
    print(mean_results)
