import numpy as np
import sqlite3
from pathlib import Path


def _detect_column_type(type_set):
    if any([val == str for val in type_set]):
        return "TEXT"
    if all([val in (int, np.int, np.uint) for val in type_set]):
        return "INT"
    if all([val in (int, np.int, np.uint, float, np.float, np.double) for val in type_set]):
        return "REAL"
    return "BLOB"


def _detect_none(column):
    if any([val is None for val in column]):
        return ""
    return " NOT NULL"


def _create_primary_key(primary_key, column_headers):
    # If primary keys in None then set to first column
    if primary_key is None:
        primary_key = column_headers[0]

    # If the the primary key is a string, then make a parenthesis around it
    if isinstance(primary_key, str):
        primary_key = "({})".format(primary_key)

    # Of Otherwise check for alternative methods
    else:
        # If primary key is an integer then use that column as key
        if isinstance(primary_key, int):
            primary_key = "({})".format(column_headers[primary_key])

        # If primary key is a list then use those elements as key
        elif isinstance(primary_key, list):
            # Convert integers to the respective columns
            primary_key = [column_headers[val] if isinstance(val, int) else val for val in primary_key]

            # If all values in the list are column-headers (which they have to be) then create the joined primary key
            if all([val in column_headers for val in primary_key]):
                primary_key = "({})".format(",".join(primary_key))

            # Otherwise throw some stuff at the user
            else:
                raise ValueError("Could not figure out primary-key in rows2sql_table(): {}".format(primary_key))
        else:
            raise ValueError("Argument 'primary_key' can not be of type {}".format(type(primary_key).__name__))

    return primary_key


def _create_create_command(column_headers, type_strs, none_strs, table_name, primary_key):
    column_specs = ",\n\t".join(["{} {}{}".format(c_name, c_type, c_none_assign)
                                 for c_name, c_type, c_none_assign in zip(column_headers, type_strs, none_strs)]) + ","
    create_command = (
            "CREATE TABLE {} (\n\t".format(table_name) +
            column_specs +
            "\n\tPRIMARY KEY {}".format(primary_key) +
            "\n)"
    )

    return create_command


def _create_insert_command(table_name, column_headers, ):
    return (
        "INSERT INTO {} ".format(table_name) +
        "({}) ".format(",".join(column_headers)) +
        "VALUES ({})".format(",".join(["?" for _ in range(len(column_headers))]))
    )


def rows2sql_table(data, database_path, table_name=None, column_headers=None, primary_key=None,
                   data_is_columns=False):
    """
    Creates a table with some data in an sqlite-database.
    If the database does not exist then it is created.
    If the table already exists then it is overwritten.

    :param list[list] data: The data itself can be all kinds of stuff as long as the elements are packed into a list of lists.
        If data_is_columns == False (default) then data should be a list of rows.
        If data_is_columns == True then data should be a list of columns.
    :param Path database_path: Location to put database.
    :param str table_name: Name of table (default is 'data_table').
        FYI 'table' is not allowed as it is a keyword in sql.
    :param list[str] column_headers: The headers of the columns in the database.
        Defaults to 'column_0', 'column_1', etc.
    :param str | int | list[int|str] primary_key: Assigns a primary key to one or some of the columns.
        str: Use the column of this name as primary key.
        int: Use this column number as primary key.
        list: Use this combination of columns as primary key, where each column is fetched wither by name or number.
    :param bool data_is_columns: Determines whether the input data is a list of rows or a list of columns. 
    """
    # Get columns or rows depending on input format
    if data_is_columns:
        columns = data
        rows = list(zip(*data))
    else:
        rows = data
        columns = list(zip(*data))

    # Detect types and non-options
    type_strs = [_detect_column_type(set([type(val) for val in column if val is not None])) for column in columns]
    none_strs = [_detect_none(column) for column in columns]

    # Default table_name
    table_name = table_name if table_name is not None else "data_table"

    # Create default headers if needed
    if column_headers is None:
        column_headers = ["column_{}".format(val) for val in range(len(columns))]
    assert all([isinstance(val, str) for val in column_headers]), "column_headers must be a list of str (or None)."

    # Get primary key string
    primary_key = _create_primary_key(primary_key, column_headers)

    # Create-command
    create_command = _create_create_command(
        column_headers=column_headers,
        type_strs=type_strs,
        none_strs=none_strs,
        table_name=table_name,
        primary_key=primary_key
    )

    # Insert-command
    insert_command = _create_insert_command(table_name=table_name, column_headers=column_headers)

    # Make connection and cursor
    connection = sqlite3.connect(str(database_path))
    cursor = connection.cursor()

    try:
        # Ensure table is not already there (overwrite anything in the way)
        cursor.execute("DROP TABLE IF EXISTS {}".format(table_name))

        # Create table
        cursor.execute(create_command)

        # Insert data
        cursor.executemany(insert_command, rows)

        # Commit changes
        connection.commit()

    finally:
        # Close
        cursor.close()
        connection.close()
