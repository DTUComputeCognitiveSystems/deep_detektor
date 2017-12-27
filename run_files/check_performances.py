from pathlib import Path
import pickle

from pandas import DataFrame, Series

from project_paths import ProjectPaths

# Results path
main_dir = Path(ProjectPaths.results, "single_train")
# main_dir = Path(ProjectPaths.results, "model_comparison")

# Get paths
paths = list(main_dir.glob("*"))

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

    except FileNotFoundError:
        pass

    # Attempt to get training settings
    try:
        settings = pickle.load(Path(path, "settings.p").open("rb"))  # type: dict

        table_row_names.append("test_programs")
        table_row_data.append(settings.get("test_programs", None))

        table_row_names.append("training_programs")
        table_row_data.append(settings.get("training_programs", None))

    except FileNotFoundError:
        pass

    ####################

    # Create series
    series = Series(
        data=table_row_data, index=table_row_names, name=path.name
    )

    # Add row
    table = table.append(series)

# Print table
print(table.to_string())
