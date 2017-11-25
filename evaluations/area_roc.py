import matplotlib.pyplot as plt
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


def plot_roc(tp_rate, fp_rate, title=""):
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.plot(fp_rate, tp_rate)
    plt.xlabel("False Negative Rate")
    plt.ylabel("False Positive Rate")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(title)
    plt.text(x=0.99,
             y=0.01,
             s="Area under ROC: {:.2f}".format(sum(tp_rate) / len(tp_rate)),
             ha="right",
             va="bottom")


if __name__ == "__main__":
    evaluation_function = AreaUnderROC()

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

    column_width = 40
    header_formatter = "{{:{0}s}} | {{:{0}s}} | {{}}".format(column_width)
    row_formatter = "{{:{0}s}} | {{:{0}s}} | {{:.2f}}".format(column_width)
    if do_print:
        print(header_formatter.format("True values", "Predictions", "Area under ROC"))
    for y_true, y_pred in tests:
        c_evaluation = evaluation_function(y_true, y_pred)
        if do_print:
            print(row_formatter.format(str(y_true), str(y_pred), c_evaluation))
