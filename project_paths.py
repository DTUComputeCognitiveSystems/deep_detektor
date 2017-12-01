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

    # Data for tensor-provider (for models)
    tensor_provider = Path(deep_fact_dir, "tensor_provider")

    # NLP-data from polyglot
    nlp_data_dir = Path(tensor_provider, "nlp_data")

    # Fast-text
    fast_text_dir = Path(tensor_provider, "fasttext")

    # Character embedding from auto-encoding
    speller_dir = Path(tensor_provider, "spelling_model")
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
