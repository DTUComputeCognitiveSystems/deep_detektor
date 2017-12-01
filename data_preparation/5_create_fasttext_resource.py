import json
import json
import re
import sqlite3
from collections import Counter
from io import TextIOWrapper
from pathlib import Path
from tempfile import NamedTemporaryFile

import fasttext
import numpy as np
import shutil

from project_paths import ProjectPaths
from util.utilities import ensure_folder

# Paths
database_path = Path(ProjectPaths.nlp_data_dir, "nlp_data.db")
storage_folder = ProjectPaths.fast_text_dir
out_path = Path(storage_folder, "vectors.db")
ensure_folder(storage_folder)

print("Creating temporary file.")
temp_file = NamedTemporaryFile(mode="w+", delete=False)  # type: TextIOWrapper
print("Temporary file at: {}".format(temp_file.name))

print("Loading all programs' tokens")
connection = sqlite3.connect(str(database_path))
cursor = connection.cursor()
rows = cursor.execute("SELECT tokens FROM tagger").fetchall()

print("Loading programs into temporary file")
for sentence in rows:
    sentence = json.loads(sentence[0])
    for token in sentence:  # type: str
        temp_file.write(" " + token)

temp_file.seek(0)
print("Make fast-text model.")
model = fasttext.skipgram(temp_file.name, 'model')


def check_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


# Connect to new database
if out_path.is_file():
    out_path.unlink()
connection = sqlite3.connect(str(out_path))
cursor = connection.cursor()

# Make tagger-table
cursor.execute(
    "CREATE TABLE embeddings ("
    "token_id INTEGER NOT NULL,"
    "token TEXT NOT NULL,"
    "vector TEXT NOT NULL,"
    "PRIMARY KEY (token_id)"
    ")"
)

print("Re-loading vectors to create DB-file")
word_pattern = re.compile("^([\S]+) ")
vector_lengths = Counter()
rows = []
token_id = 0
with Path("model.vec").open("r") as in_file:
    # Skip header
    next(in_file)

    # Go through embeddings
    embeddings = dict()
    for line in in_file:
        word_search = word_pattern.search(line)
        word = word_search.group(1)
        vector = [float(val) for val in line[word_search.end(0):].split(" ") if check_float(val)]
        vector_lengths[len(vector)] += 1
        embeddings[word] = vector

    # Ensure sorted file
    for word in sorted(embeddings.keys()):
        rows.append([token_id, word, json.dumps(embeddings[word])])
        token_id += 1

insert_command = "INSERT INTO embeddings (token_id, token, vector)" \
                 " VALUES (?, ?, ?)"
cursor.executemany(insert_command, rows)
connection.commit()
cursor.close()
connection.close()

print("Moving model-files.")
files = [Path("model.bin"), Path("model.vec")]
for file_path in files:
    shutil.move(str(file_path), str(Path(out_path.parent, file_path.name)))

print("Testing loading of vectors.")
connection = sqlite3.connect(str(out_path))
cursor = connection.cursor()
rows = cursor.execute("SELECT token, vector FROM embeddings").fetchall()
word_embeddings = dict()
for row in rows:
    word_embeddings[row[0]] = np.array(eval(row[1]))

assert len(vector_lengths) == 1, "More than one length of vector was observed."
print("\n" + "-"*30 + "\nDone.")
