from pathlib import Path
import pickle
import numpy as np

from pandas import DataFrame, Series
import pandas as pd
import pandas.io.formats.format as fmt

from project_paths import ProjectPaths

pd.set_option('expand_frame_repr', False)
pd.set_option('max_colwidth', 80)
pd.set_option('colheader_justify', 'left')

# Results path
# main_dir = Path(ProjectPaths.results, "single_train")
main_dir = Path(ProjectPaths.results, "final_model_comparison_backup_2")

# Get paths
paths = [path for path in main_dir.glob("*") if path.is_dir()]

# Go through paths
table = DataFrame()
for path in paths:
    table_row_data = []
    table_row_names = []

    ####################

    # Name-file
    try:
        with Path(path, "name.txt").open("r") as file:
            name = file.readline().strip()
            table_row_names.append("settings_name")
            table_row_data.append(name)
    except FileNotFoundError:
        pass

    # Attempt to get training performance
    try:
        results_train = pickle.load(Path(path, "results_train.p").open("rb"))
        if isinstance(results_train, DataFrame):
            results_train = results_train.get(results_train.keys()[0])

        fetch = ["F1", "AreaUnderROC"]
        for attr in fetch:
            table_row_names.append("tr_" + attr)
            value = results_train.get(attr)
            table_row_data.append(value if value is not None else "-")

        table_row_names.append("tr_" + "TPR")
        table_row_data.append(results_train.get("TP") / (results_train.get("TP") + results_train.get("FN")))

    except FileNotFoundError:
        pass

    # Attempt to get test performance
    try:
        results_test = pickle.load(Path(path, "results_test.p").open("rb"))
        if isinstance(results_test, DataFrame):
            results_test = results_test.get(results_test.keys()[0])

        fetch = ["F1", "AreaUnderROC"]
        for attr in fetch:
            table_row_names.append("te_" + attr)
            value = results_test.get(attr)
            table_row_data.append(value if value is not None else "-")

        table_row_names.append("te_" + "TPR")
        table_row_data.append(results_test.get("TP") / (results_test.get("TP") + results_test.get("FN")))

    except FileNotFoundError:
        pass

    # Attempt to get training settings
    try:
        settings = pickle.load(Path(path, "settings.p").open("rb"))  # type: dict

        # table_row_names.append("test_programs")
        # val = settings.get("test_programs", None)
        # if isinstance(val, (list, tuple, np.ndarray)):
        #     val = len(val)
        # table_row_data.append(val)
        #
        # table_row_names.append("training_programs")
        # val = settings.get("training_programs", None)
        # if isinstance(val, (list, tuple, np.ndarray)):
        #     val = len(val)
        # table_row_data.append(val)

    except FileNotFoundError:
        pass

    # Check if program finished
    table_row_names.append("Done")
    table_row_data.append("Done" if Path(path, "done.txt").exists() else "Not Done")

    ####################

    # Create series
    series = Series(
        data=table_row_data, index=table_row_names, name=path.name
    )

    # Add row
    table = table.append(series)



# # float_formatter = pd.get_option("display.float_format")
# # if float_formatter is None:
# #     float_formatter = '%% .%dg' % pd.get_option("display.precision")
# float_formatter = "%.2f"
#
# # Get string-columns
# str_locs = [isinstance(val, str) for val in table.loc[table.index[0]]]
# str_columns = table[np.array(table.keys())[str_locs]]
#
# # Get maximum string-lengths
# vec_len = np.vectorize(len)
# max_lengths = vec_len(str_columns).max(0)
#
# # Make string formatters
# str_formatters = {
#     key: "%<{}s".format(length) for key, length in zip(str_columns, max_lengths)
# }
#
# # All formatters
# formatters = {key: str_formatters[key] if is_str else float_formatter
#               for key, is_str in zip(table, str_locs)}
#
# formatters['Done'] = "%s"
# formatters['settings_name'] = "%s"
# formatters['te_AreaUnderROC'] = "%s"
# formatters['te_F1'] = "%s"
# formatters['test_programs'] = "%s"
# formatters['tr_AreaUnderROC'] = "%s"
# # formatters['tr_F1'] = "%s"
# formatters['training_programs'] = "%s"
#
# formatters = {key: lambda x: val % x for key, val in formatters.items()}
#
# print(table.to_string(formatters=formatters))


# Print table
print(table.to_string())

print("\n\nLatex:\n")
scores = ["AreaUnderROC", "F1", "TPR"]
rows = sorted(list(table.iterrows()))
for name, row in rows:
    row_str = name

    for val in scores:
        row_str += " & \\perfsplit{{{:.2f}}}{{{:.2f}}}"\
            .format(row.get("tr_" + val), row.get("te_" + val))
    print(row_str + "\\\\")


