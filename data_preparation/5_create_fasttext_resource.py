import json
import sqlite3
from io import TextIOWrapper
from pathlib import Path
from tempfile import NamedTemporaryFile

import fastText

from project_paths import ProjectPaths
from util.utilities import ensure_folder

# Paths
database_path = Path(ProjectPaths.nlp_data_dir, "nlp_data.db")
storage_folder = ProjectPaths.fast_text_dir
out_path = Path(storage_folder, "vectors.db")
ensure_folder(storage_folder)

print("CWD: {}".format(Path.cwd()))

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
model = fastText.train_unsupervised(
    input=temp_file.name,
    #lr=0.1,
    epoch=500,
    minCount=4,
    model="cbow",
    thread=18
)

# Save model
model.save_model(str(Path(storage_folder, "model.bin")))
