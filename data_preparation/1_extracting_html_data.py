"""
Fetches data from web-crawl and creates HTML-files that can be annotated.
Also creates a database with all programs and sentences.
"""
import sqlite3
from pathlib import Path

from data_preparation.classes.data_cleaner import DebattenDataCleaner
from data_preparation.data_preparation_utility import clean_str
from project_paths import ProjectPaths

# Create data-cleaner
data_cleaner = DebattenDataCleaner(raw_subtitles_dir=ProjectPaths.subtitles_crawl,
                                   cleaned_subtitles_dir=ProjectPaths.preannotated_dir)
data_cleaner.getFileLocation()

# Get files and program IDs
files, program_id = data_cleaner.getRawFilePaths()

print('The ten first examples are')
for path, name in zip(*data_cleaner.getRawFilePaths(subset=10)):
    print("\t", name, path)
print("")

# Clean all programs
all_programs = data_cleaner.getCleanedPrograms(files, program_id)

# Determine programs with few sentences
low_limit = 100
high_limit = 1000
check_program = []
for program in all_programs:
    if len(all_programs[program]['sentences']) < low_limit or len(all_programs[program]['sentences']) > high_limit:
        check_program.append(program)
print('\nThe following programs has less than {} sentences or more than {}.'.format(low_limit, high_limit))
for program in check_program:
    paragraphs = all_programs[program]['sentences']
    print('\tProgram {} has {} paragraphs'.format(program, len(paragraphs)))

print("\nConclusions based on inspection:")
print("\t8571627: Test program, Always invalid!")
print("\t8793533: Brexit program, rolling subtitles, can be fixed!")
print("\t8905036: rolling subs")
print("\t8573626: empty program, cant be fixed")
print("\t8975996: rolling")
print("\t9024801: rolling")

# Removing weird programs
remove_programs = ['8571627', '8793533', '8905036', '8573626', '8975996', '9024801']
print("\nFor removal: {}".format(remove_programs))
print("\tNumber of programs before: {}".format(len(all_programs)))
for program in remove_programs:
    del all_programs[program]
print("\tNumber of programs after: {}".format(len(all_programs)))


# Export data to disk
print("\nWriting specific programs to html-files.")
data_cleaner.export_programs(all_programs)

print("\nCreating database for all programs")
print("\tRemoving pre-existing database.")
database_path = Path(ProjectPaths.tensor_provider, "all_programs.db")
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
    "start_time TEXT NOT NULL," 
    "end_time TEXT NOT NULL," 
    "PRIMARY KEY (program_id, sentence_id)"  
    ")"
)

print("\tCreating rows.")
all_data = []
for program_id in all_programs.keys():
    start_times = all_programs[program_id]['start time']
    end_times = all_programs[program_id]['end time']
    sentences = all_programs[program_id]['sentences']
    program_id = int(program_id)

    for sentence_nr, (start_time, end_time, sentence) in enumerate(zip(start_times, end_times, sentences)):
        all_data.append([program_id, sentence_nr+1, clean_str(sentence), start_time, end_time])

print("\tInserting rows")
insert_command = "INSERT INTO programs (program_id, sentence_id, sentence, start_time, end_time)" \
                 " VALUES (?, ?, ?, ?, ?)"
cursor.executemany(insert_command, all_data)

print("\tCommitting and closing.")
connection.commit()
cursor.close()
connection.close()

print("\n" + "-" * 40 + "\nDone.")
