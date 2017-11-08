from pathlib import Path

# Main data-directory
data_dir = Path("..", "data")

# Two main directories for data
deep_fact_dir = Path(data_dir, "DeepFactData")
dr_detektor_automatic_fact_checking_dir = Path(data_dir, "DRDetektorAutomaticFactChecking")

# Data from data-cleaning
data_matrix_path = Path(deep_fact_dir, "annotated", "data_matrix_sample_programs.csv")

# NLP-data from polyglot
nlp_data_dir = Path(deep_fact_dir, "nlp_data")
embeddings_file = Path(nlp_data_dir, "embeddings.csv")
pos_tags_file = Path(nlp_data_dir, "pos_tags.csv")

# Character embedding from auto-encoding
speller_dir = Path(deep_fact_dir, "spelling_model")
speller_results_file = Path(speller_dir, "results.json")
speller_char_vocab_file = Path(speller_dir, "char_embedding.json")
speller_translator_file = Path(speller_dir, "string_translator.json")
speller_encoder_checkpoint_file = Path(speller_dir, "checkpoint", "speller_encode.ckpt")
