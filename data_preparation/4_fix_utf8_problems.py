import json
import sqlite3
from pathlib import Path

from project_paths import ProjectPaths

# Paths
data_dir = ProjectPaths.nlp_data_dir
database_path = Path(data_dir, "nlp_data_ubuntu.db")
output_path = Path(data_dir, "nlp_data.db")

print("Loading all programs' tokens")
connection = sqlite3.connect(str(database_path))
cursor = connection.cursor()
rows = cursor.execute("SELECT program_id, sentence_id, pos, tokens FROM tagger").fetchall()
cursor.close()
connection.close()

# Clean rows
new_rows = []
for program_id, sentence_id, pos, tokens in rows:
    tokens = json.dumps(json.loads(tokens), ensure_ascii=False)
    new_rows.append((program_id, sentence_id, pos, tokens))


# Connect to new database
if output_path.is_file():
    output_path.unlink()
connection = sqlite3.connect(str(output_path))
cursor = connection.cursor()

# Make tagger-table
cursor.execute(
    "CREATE TABLE tagger ("
    "program_id INTEGER NOT NULL,"
    "sentence_id INTEGER NOT NULL,"
    "pos TEXT NOT NULL,"
    "tokens TEXT NOT NULL,"
    "PRIMARY KEY (program_id, sentence_id)"
    ")"
)

# Insert all
insert_command = "INSERT INTO tagger (program_id, sentence_id, pos, tokens)" \
                 " VALUES (?, ?, ?, ?)"
cursor.executemany(insert_command, new_rows)
connection.commit()

