"""
Takes annotated data from DR and creates dataset.
"""
import re
import sqlite3
from collections import Counter
from pathlib import Path

from editdistance import eval as editdistance

from data_preparation.classes.annotated_data_cleaner import DebattenAnnotatedDataCleaner
from project_paths import ProjectPaths

# Set paths
annotated_data_dir = ProjectPaths.dr_annotated_subtitles_dir  # Path where DR stores annotated data.
storage_dir = ProjectPaths.annotated_data_dir

# Open cleaner in directory
annotatedData = DebattenAnnotatedDataCleaner(annotated_data_dir)
file_paths = annotatedData.getFilePaths()

# Get data and labels of annotated programs
data, labels = annotatedData.getAllCleanedProgramSentences(disp=True)

# Number of observations
N = len(data)

# Conversion from file-name to program ID (manually inspected in databases)
program_name2id = dict(
    program1=7308025,
    program2=2294023,
    program3=2315222,
    program4=2337314,
    program5=2359717,
    program6=2304494,
    program7=2348260,
    program8=3411204,
    program9=3570949,
    program10=3662558
)

# Program 2337314 sentence 1 is incorrect in annotated dataset !


#################################################################
# annotated_programs.db

# Prepare data for database (strings and removing single-word claims)
pattern = re.compile("^[\S]+$")
database_data = [
    [program_name2id[row[0]], row[1], row[2], str(row[4]), str(row[3]), row[4] is not None]
    for row in data
    if not pattern.match(str(row[4])) or row[4] is None
]

print("\nCreating database for all programs")
print("\tRemoving pre-existing database.")
database_path = Path(storage_dir, "annotated_programs.db")
if database_path.is_file():
    database_path.unlink()

print("\tConnection")
connection = sqlite3.connect(str(database_path))
cursor = connection.cursor()

print("\tCreating table")
cursor.execute(
    "CREATE TABLE programs ("
    "program_id INTEGER NOT NULL,"
    "sentence_id INTEGER NOT NULL,"
    "sentence TEXT NOT NULL,"
    "claim TEXT,"
    "claim_idx TEXT,"
    "claim_flag INTEGER NOT NULL,"
    "PRIMARY KEY (program_id, sentence_id)"
    ")"
)

print("\tInserting rows")
insert_command = "INSERT INTO programs (program_id, sentence_id, sentence, claim, claim_idx, claim_flag)" \
                 " VALUES (?, ?, ?, ?, ?, ?)"
cursor.executemany(insert_command, database_data)

print("\tCommitting and closing.")
connection.commit()
cursor.close()
connection.close()

# TODO: Detection of leading and trailing spaces may have been removed - were they necessary at this point?


#################################################################
# Inspection database

# Data from annotated dataset
annotated_data_sentences = {(row[0], row[1]): row[2] for row in database_data}
n_sentences_in_annotated_programs = Counter()
for row in database_data:
    n_sentences_in_annotated_programs[row[0]] += 1

# Data from web-crawl
connection = sqlite3.connect(str(Path(ProjectPaths.preannotated_dir, "all_programs.db")))
cursor = connection.cursor()
cursor.execute("SELECT program_id, sentence_id, sentence FROM programs")
crawl_data = cursor.fetchall()
cursor.close()
connection.close()
crawl_data_sentences = {(row[0], row[1]): row[2] for row in crawl_data if row[0] in n_sentences_in_annotated_programs}
n_sentences_in_crawl_programs = Counter()
for row in crawl_data_sentences:
    if row[0] in n_sentences_in_annotated_programs:
        n_sentences_in_crawl_programs[row[0]] += 1

# Create inspection database
inspection_path = Path(storage_dir, "inspection_programs.db")
if inspection_path.is_file():
    inspection_path.unlink()
connection = sqlite3.connect(str(inspection_path))
cursor = connection.cursor()
cursor.execute(
    "CREATE TABLE programs ("
    "program_id INTEGER NOT NULL,"
    "sentence_id INTEGER NOT NULL,"
    "crawl_sentence TEXT NOT NULL,"
    "annotated_sentence TEXT NOT NULL,"
    "edit_distance INTEGER,"
    "overlap REAL,"
    "PRIMARY KEY (program_id, sentence_id)"
    ")"
)

# Make inspection-values
rows = []
for program_id in n_sentences_in_annotated_programs.keys():
    sentence_id = 0
    while True:
        sentence_id += 1
        key = (program_id, sentence_id)

        if key not in annotated_data_sentences and key not in crawl_data_sentences:
            break

        annotated_sentence = annotated_data_sentences.get(key, "")
        crawl_sentence = crawl_data_sentences.get(key, "")

        distance = editdistance(annotated_sentence, crawl_sentence)

        if crawl_sentence:
            relative_distance = (len(crawl_sentence) - distance) / len(crawl_sentence)
        else:
            relative_distance = None

        rows.append([program_id, sentence_id, crawl_sentence, annotated_sentence, distance, relative_distance])


insert_command = "INSERT INTO programs (program_id, sentence_id, crawl_sentence, " \
                 "annotated_sentence, edit_distance, overlap)" \
                 " VALUES (?, ?, ?, ?, ?, ?)"
cursor.executemany(insert_command, rows)

connection.commit()
cursor.close()
connection.close()
