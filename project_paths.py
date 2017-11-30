from pathlib import Path


class ProjectPaths:

    # Main data-directory
    data_dir = Path("..", "data")

    # Two main directories for data
    deep_fact_dir = Path(data_dir, "DeepFactData")
    dr_detektor_automatic_fact_checking_dir = Path(data_dir, "DRDetektorAutomaticFactChecking")

    # Data from annotation task (from DR)
    dr_annotated_subtitles_dir = Path(dr_detektor_automatic_fact_checking_dir, "annotatorProgram", "annotatedSubtitles")
    subtitles_crawl = Path(deep_fact_dir, "subtitles crawl")

    # Data from data-cleaning
    preannotated_dir = Path(deep_fact_dir, "preannotated")
    annotated_data_dir = Path(deep_fact_dir, "annotated")
    data_matrix_path = Path(annotated_data_dir, "data_matrix_sample_programs.csv")
    all_programs = Path(preannotated_dir, "all_programs.pickle")

    # NLP-data from polyglot
    nlp_data_dir = Path(deep_fact_dir, "nlp_data")
    # embeddings_file = Path(nlp_data_dir, "embeddings.csv")  # No longer in use
    pos_tags_file = Path(nlp_data_dir, "pos_tags.csv")

    # Fast-text
    fast_text_dir = Path(deep_fact_dir, "fasttext")
    embeddings_file = Path(fast_text_dir, "vectors.csv")

    # Character embedding from auto-encoding
    speller_dir = Path(deep_fact_dir, "spelling_model")
    speller_results_file = Path(speller_dir, "results.json")
    speller_char_vocab_file = Path(speller_dir, "char_embedding.json")
    speller_translator_file = Path(speller_dir, "string_translator.json")
    speller_encoder_checkpoint_file = Path(speller_dir, "checkpoint", "speller_encode.ckpt")

    # Directory for outputs (evaluations, statistics etc.)
    results = Path(data_dir, "results")

    # Get all paths for quick changing of working directory
    __path_names = [name for name in sorted(locals().keys()) if not name.startswith("__")]
    __paths_moved = False

    @classmethod
    def set_path_to_repository(cls, path):
        if not cls.__paths_moved:
            cls.__paths_moved = True
            for name in cls.__path_names:
                new_path = Path(path, getattr(cls, name))
                setattr(cls, name, new_path)

    @classmethod
    def get_names(cls):
        names = list(sorted(cls.__path_names))
        for name in names:
            print(name)

    @classmethod
    def get_paths(cls):
        return list(sorted([getattr(cls, name) for name in cls.__path_names]))

    @classmethod
    def print_paths(cls):
        max_name_length = max([len(name) for name in cls.__path_names])
        formatter = "{{:{}s}} : {{}}".format(max_name_length)
        path_and_names = list(sorted([(getattr(cls, name), name) for name in cls.__path_names]))
        for path, name in path_and_names:
            print(formatter.format(name, path))
