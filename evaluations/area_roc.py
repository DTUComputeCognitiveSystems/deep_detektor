from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from evaluations import Evaluation, DataPositives, DataNegatives


class AreaUnderROC(Evaluation):
    def __call__(self, y_true, y_pred, y_pred_binary=None):
        # Get ROC
        tp_rate, fp_rate = make_roc(y_true, y_pred, y_pred_binary)

        # Compute area
        area = sum(tp_rate) / len(tp_rate)

        return area

    def name(self):
        return "AreaUnderROC"


class ROC(Evaluation):
    @property
    def is_single_value(self):
        return False

    def __call__(self, y_true, y_pred, y_pred_binary=None):
        return make_roc(y_true, y_pred, y_pred_binary)

    def name(self):
        return "ROC"


def make_roc(y_true, y_pred, y_pred_binary):
    # Count positives and negatives
    n_positives = DataPositives()(y_true=y_true,
                                  y_pred=y_pred,
                                  y_pred_binary=y_pred_binary)
    n_negatives = DataNegatives()(y_true=y_true,
                                  y_pred=y_pred,
                                  y_pred_binary=y_pred_binary)

    # Sort true classes with evaluation
    evaluations = [(prediction_val, true_val) for true_val, prediction_val
                   in zip(y_true, y_pred)]
    evaluations = list(sorted(evaluations, key=lambda x: -x[0]))

    # Compute TP-rate and FP-rate
    positives_delta = 1 / n_positives
    negatives_delta = 1 / n_negatives
    tp_rate = [0]
    fp_rate = [0]
    for _, new_positive in evaluations:

        if new_positive:
            tp_rate[-1] += positives_delta
        else:
            fp_rate.append(fp_rate[-1] + negatives_delta)
            tp_rate.append(tp_rate[-1])

    return tp_rate, fp_rate


def mean_rocs(curves, n_bins=101):
    bins = defaultdict(lambda: [])
    delta = 1. / (n_bins - 1)
    delimiters = list(reversed([delta * val for val in range(n_bins)]))
    delimiters[0] = 1.1

    # For each curve append all values that fit into bins
    for nr, curve in enumerate(curves):
        c_limits = list(delimiters)
        c_limit = c_limits.pop()
        c_bin = 0
        for tp, fp in zip(*curve):
            while fp > c_limit:
                c_bin += 1
                c_limit = c_limits.pop()

            bins[c_bin].append(tp)

    # Mean bins
    for key in bins.keys():
        bins[key] = np.mean(bins[key])

    # Make points ready for plotting
    tp_rate = []
    fp_rate = []
    for val, limit in enumerate(reversed(delimiters)):
        fp_rate.append(limit)
        if val in bins:
            tp_rate.append(bins[val])
        else:
            tp_rate.append(tp_rate[-1])

    return tp_rate, fp_rate


def plot_roc(tp_rate, fp_rate,
             title=None, label="", color=None, line_width=None, linestyle=None,
             corner_text=True, center_line=True):
    if center_line:
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--")

    options = dict()
    if color is not None:
        options["color"] = color
    if line_width is not None:
        options["line_width"] = line_width
    if linestyle is not None:
        options["linestyle"] = linestyle
    plt.plot(fp_rate, tp_rate, label=label, **options)
    plt.xlabel("False Negative Rate")
    plt.ylabel("False Positive Rate")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    if title is not None:
        plt.title(title)
    if corner_text:
        plt.text(x=0.99,
                 y=0.01,
                 s="Area under ROC: {:.2f}".format(sum(tp_rate) / len(tp_rate)),
                 ha="right",
                 va="bottom")


def plot_multiple_rocs(rocs, labels=None, title=None, center_line=True):
    labels = ["" for _ in rocs] if labels is None else labels
    first_run = True
    for roc, label in zip(rocs, labels):
        if first_run or not center_line:
            plot_roc(*roc, label=label, corner_text=False)
        else:
            plot_roc(*roc, label=label, corner_text=False, center_line=False)
    if title is not None:
       plt.title(title)


if __name__ == "__main__":
    plt.close("all")

    tests = [
        ([0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 0, 1]),
        ([0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 1, 1]),
        ([0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]),
        ([0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 1]),
        ([0, 0, 0, 1, 1, 1], [0, 1, 1, 1, 1, 1]),
        ([0, 0, 0, 1, 1, 1], [1, 1, 1, 1, 1, 1]),
        #
        ([0, 0, 0, 1, 1, 1], [0, 0, 0, 0.7, 0.8, 0.9]),
        ([0, 0, 0, 1, 1, 1], [0, 0, 0.75, 0.7, 0.8, 0.9]),
        ([0, 0, 0, 1, 1, 1], [0, 0.76, 0.75, 0.7, 0.8, 0.9]),
        ([0, 0, 0, 1, 1, 1], [0, 0.86, 0.75, 0.7, 0.8, 0.9]),
    ]

    do_print = True

    # Test area under ROC
    evaluation_function = AreaUnderROC()
    column_width = 40
    header_formatter = "{{:{0}s}} | {{:{0}s}} | {{}}".format(column_width)
    row_formatter = "{{:{0}s}} | {{:{0}s}} | {{:.2f}}".format(column_width)
    if do_print:
        print(header_formatter.format("True values", "Predictions", "Area under ROC"))
    for y_true, y_pred in tests:
        c_evaluation = evaluation_function(y_true, y_pred)
        if do_print:
            print(row_formatter.format(str(y_true), str(y_pred), c_evaluation))

    # Test ROC and plotting
    evaluation_function = ROC()
    rocs = []
    labels = []
    test_nr = 0
    for y_true, y_pred in tests:
        c_roc = evaluation_function(y_true, y_pred)
        rocs.append(c_roc)
        labels.append("Test {}".format(test_nr))
        test_nr += 1
    plot_multiple_rocs(rocs=rocs, labels=labels, title="All ROCs")
    mean = mean_rocs(rocs)
    plot_roc(*mean, label="Mean", color="black", center_line=False)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

