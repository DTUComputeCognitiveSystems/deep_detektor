import json
import random
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Callable

import fastText
import matplotlib.pyplot as plt
import nltk
import numpy as np
import tensorflow as tf

from models.speller.recurrent_speller import RecurrentSpeller
from project_paths import ProjectPaths
from util.utilities import save_fig, ensure_folder, redirect_stdout_to_file, close_stdout_file


class TensorProvider:
    def __init__(self, verbose=False, end_of_word_char="$", fill=0):
        self.fill = fill

        # Make graph and session
        self._tf_graph = tf.Graph()
        self._sess = tf.Session(graph=self._tf_graph)

        ###################
        # Character embedding (auto-encoder)

        if verbose:
            print("Loading character embedding.")
        with ProjectPaths.speller_char_vocab_file.open("r") as file:
            self.char_embedding = json.load(file)
        with ProjectPaths.speller_results_file.open("r") as file:
            speller_results = json.load(file)
        with ProjectPaths.speller_translator_file.open("r") as file:
            self.string_translator = {int(val[0]): val[1] for val in json.load(file).items()}

        with self._tf_graph.as_default():
            self.char_embedding_size = speller_results['cells']
            self.recurrent_speller = RecurrentSpeller(n_inputs=len(self.char_embedding),
                                                      n_outputs=len(self.char_embedding),
                                                      n_encoding_cells=self.char_embedding_size,
                                                      n_decoding_cells=self.char_embedding_size,
                                                      character_embedding=self.char_embedding,
                                                      end_of_word_char=end_of_word_char)
            self.recurrent_speller.load_encoder(sess=self._sess, file_path=ProjectPaths.speller_encoder_checkpoint_file)

        ###################
        # Keys of all data

        if verbose:
            print("Loading keys.")
        self.labels = dict()
        self._keys = []
        program_ids = set()
        database_path = Path(ProjectPaths.tensor_provider, "all_programs.db")
        connection = sqlite3.connect(str(database_path))
        cursor = connection.cursor()
        rows = cursor.execute("SELECT program_id, sentence_id FROM programs").fetchall()
        for row in rows:
            key = (row[0], row[1])
            program_ids.add(row[0])
            self._keys.append(key)
            self.labels[key] = None
        cursor.close()
        connection.close()

        ###################
        # Labels of original data

        if verbose:
            print("Loading labels.")
        annotated_program_ids = set()
        annotated_keys = set()
        database_path = Path(ProjectPaths.tensor_provider, "annotated_programs.db")
        connection = sqlite3.connect(str(database_path))
        cursor = connection.cursor()
        rows = cursor.execute("SELECT program_id, sentence_id, claim_flag FROM programs").fetchall()
        for row in rows:
            key = (row[0], row[1])
            annotated_program_ids.add(row[0])
            annotated_keys.add(key)
            self.labels[key] = bool(row[2])
        cursor.close()
        connection.close()

        ###################
        # Get annotated and non-annotated data

        self._program_ids = program_ids
        self._annotated_program_ids = annotated_program_ids
        self._annotated_keys = annotated_keys

        ###################
        # Word embeddings

        # Load fasttext-model
        if verbose:
            print("Loading fastText.")
        self.word_embeddings = fastText.load_model(str(Path(ProjectPaths.fast_text_dir, 'model.bin')))
        self.word_embedding_size = int(self.word_embeddings.get_dimension())

        ###################
        # Tokenized texts and POS-tags

        if verbose:
            print("Loading POS-taggings and tokenized elements.")
        self.pos_tags = dict()
        self.tokens = dict()
        self.pos_vocabulary = set()
        self.vocabulary = set()

        database_path = Path(ProjectPaths.nlp_data_dir, "nlp_data.db")
        connection = sqlite3.connect(str(database_path))
        cursor = connection.cursor()
        rows = cursor.execute("SELECT program_id, sentence_id, pos, tokens FROM tagger").fetchall()
        for row in rows:
            key = (row[0], row[1])
            pos_tags = json.loads(row[2])
            tokens = json.loads(row[3])

            self.pos_vocabulary.update(pos_tags)
            self.vocabulary.update(tokens)
            self.pos_tags[key] = pos_tags
            self.tokens[key] = tokens
        self.pos_vocabulary = {val: idx for idx, val in enumerate(sorted(list(self.pos_vocabulary)))}
        self.vocabulary = {val: idx for idx, val in enumerate(sorted(list(self.vocabulary)))}
        self.pos_embedding_size = len(self.pos_vocabulary)
        self._pos_embedding = np.eye(len(self.pos_vocabulary), len(self.pos_vocabulary))

        ###################
        # BOW-settings

        # # Set an transformer for BOW (ex. stemmer)
        stemmer = nltk.stem.SnowballStemmer('danish')
        self.bow_transformer = lambda x: stemmer.stem(x.lower())  # type: Callable

        # The set of known words
        if self.bow_transformer is None:
            self._complete_bow_vocabulary = self.vocabulary
        else:
            bow_vocabulary = {self.bow_transformer(word) for word in self.vocabulary.keys()}
            self._complete_bow_vocabulary = {word: idx for idx, word in enumerate(sorted(list(bow_vocabulary)))}
        self.bow_vocabulary = self._complete_bow_vocabulary

    def set_bow_vocabulary(self, vocabulary=None):
        self.bow_vocabulary = self._complete_bow_vocabulary if vocabulary is None else vocabulary

    ####
    # Program IDs

    @property
    def off_limits_programs(self):
        return 8720741, 9284846, 8665813

    def program_ids(self, access_restricted_data=False):
        if not access_restricted_data:
            raise ValueError("You are not allowed to access this data!!")
        return self._program_ids

    @property
    def accessible_program_ids(self):
        return [val for val in self._program_ids if val not in self.off_limits_programs]

    def annotated_program_ids(self, access_restricted_data=False):
        if not access_restricted_data:
            raise ValueError("You are not allowed to access this data!!")
        return self._annotated_program_ids

    @property
    def accessible_annotated_program_ids(self):
        return [val for val in self._annotated_program_ids if val not in self.off_limits_programs]

    ####
    # Keys

    def keys(self, access_restricted_data=False):
        if not access_restricted_data:
            raise ValueError("You are not allowed to access this data!!")
        return self._keys

    @property
    def accessible_keys(self):
        return [val for val in self._keys if val[0] not in self.off_limits_programs]

    def annotated_keys(self, access_restricted_data=False):
        if not access_restricted_data:
            raise ValueError("You are not allowed to access this data!!")
        return self._annotated_keys

    @property
    def accessible_annotated_keys(self):
        return [val for val in self._annotated_keys if val[0] not in self.off_limits_programs]

    def _get_word_embeddings(self, tokens, n_words=None):

        # Determine longest sentence (in word-count)
        if n_words is None:
            n_words = max([len(val) for val in tokens])

        # Initialize array
        out_array = np.full((len(tokens), n_words, self.word_embedding_size), self.fill, dtype=np.float64)

        # Insert word-vectors
        for text_nr, text in enumerate(tokens):
            for word_nr, word in enumerate(text):
                out_array[text_nr, word_nr, :] = self.word_embeddings.get_word_vector(word)

        # Return
        return out_array

    def _get_word_embeddings_sum(self, tokens, do_mean=False):

        # Initialize array
        out_array = self._get_word_embeddings(tokens=tokens)

        # Compute sum
        out_array = out_array.sum(axis=1)

        # Compute means if wanted
        if do_mean:
            lengths = np.array([len(val) for val in tokens])
            out_array /= lengths

        # Return
        return out_array

    def _get_pos_tags(self, data_keys, n_words):
        # Initialize array
        out_array = np.full((len(data_keys), n_words, len(self.pos_vocabulary)), self.fill)

        # Insert pos-tags
        for key_nr, key in enumerate(data_keys):
            c_pos_tags = np.array([self.pos_vocabulary[val] for val in self.pos_tags[key]])

            out_array[key_nr, :c_pos_tags.shape[0], :] = self._pos_embedding[:, c_pos_tags].T

        return out_array

    def _get_character_embedding(self, data_keys, n_words):
        # Initialize array
        out_array = np.full((len(data_keys), n_words, self.char_embedding_size), self.fill, dtype=np.float64)

        # Go through elements
        for key_nr, key in enumerate(data_keys):
            c_words = self.tokens[key]  # type: list
            c_words = [val.translate(self.string_translator) for val in c_words]
            c_words = ["".join([char for char in word if char in self.char_embedding]) for word in c_words]

            # Run through speller
            temp = self.recurrent_speller.get_encoding(self._sess, c_words)

            # Insert
            out_array[key_nr, :len(c_words), :] = temp

        return out_array

    def _get_bow_tensors(self, tokens):

        # Transform data if wanted
        if self.bow_transformer is not None:
            tokens = [[self.bow_transformer(word) for word in text] for text in tokens]

        # Initialize array
        out_array = np.full((len(tokens), len(self.bow_vocabulary)), 0)

        # Go through texts and words
        for text_nr, text in enumerate(tokens):
            text_bow = Counter()
            text_bow.update([self.bow_vocabulary[word] for word in text if word in self.bow_vocabulary])
            word_idxs, word_counts = zip(*text_bow.items())

            out_array[text_nr, word_idxs] = word_counts

        return out_array

    def extract_programs_vocabulary(self, train_keys_or_idxs):
        train_keys_or_idxs = self._convert_to_keys(train_keys_or_idxs)

        # Get tokens of dataset
        tokens = [self.tokens[val] for val in train_keys_or_idxs]

        # Make vocabulary of dataset
        vocabulary = set()
        for text in tokens:
            vocabulary.update(text)

        # Make vocabulary
        bow_vocabulary = {self.bow_transformer(word) for word in vocabulary}
        bow_vocabulary = {word: idx for idx, word in enumerate(sorted(list(bow_vocabulary)))}

        return bow_vocabulary

    def _convert_to_keys(self, data_keys_or_idxs):
        if isinstance(data_keys_or_idxs[0], (int, np.int32, np.int64)):
            data_keys_or_idxs = [self._keys[val] for val in data_keys_or_idxs]

        return data_keys_or_idxs

    def _get_labels(self, data_keys):
        return np.array([self.labels[key] for key in data_keys])

    def input_dimensions(self, word_embedding=False, pos_tags=False, char_embedding=False,
                         bow=False, embedding_sum=False):
        # Check consistency
        sequential_data = any([word_embedding, pos_tags, char_embedding])
        static_data = any([bow, embedding_sum])
        assert not (static_data and sequential_data), "Sequential data and static data can not be mixed (yet)"

        d = 0

        if word_embedding:
            d += self.word_embedding_size

        if pos_tags:
            d += self.pos_embedding_size

        if char_embedding:
            d += self.char_embedding_size

        if bow:
            d += len(self.bow_vocabulary)

        if embedding_sum:
            d += self.word_embedding_size

        return d

    def load_concat_input_tensors(self, data_keys_or_idx,
                                  word_embedding=False, word_embedding_success=False,
                                  pos_tags=False, char_embedding=False,
                                  bow=False, embedding_sum=False):
        # Check consistency
        sequential_data = any([word_embedding, pos_tags, char_embedding])
        static_data = any([bow, embedding_sum])
        assert not (static_data and sequential_data), "Sequential data and static data can not be mixed (yet)"

        data = self.load_data_tensors(data_keys_or_idx=data_keys_or_idx,
                                      word_embedding=word_embedding,
                                      word_embedding_success=word_embedding_success,
                                      pos_tags=pos_tags,
                                      char_embedding=char_embedding,
                                      bow=bow,
                                      embedding_sum=embedding_sum)

        tensors = []

        if word_embedding:
            tensors.append(data["word_embedding"])

        if pos_tags:
            tensors.append(data["pos_tags"])

        if char_embedding:
            tensors.append(data["char_embedding"])

        if bow:
            tensors.append(data["bow"])

        if embedding_sum:
            tensors.append(data["embedding_sum"])

        if sequential_data:
            concatenated = np.concatenate(tensors, axis=2)
        else:
            concatenated = np.concatenate(tensors, axis=1)

        return concatenated

    def load_labels(self, data_keys_or_idx, convert_none_to_false=True):
        data = self.load_data_tensors(data_keys_or_idx=data_keys_or_idx,
                                      labels=True)

        labels = data["labels"]
        if convert_none_to_false:
            labels = np.array([val if val is not None else False for val in labels])

        return labels

    def load_tokens(self, data_keys_or_idx):
        data_keys = self._convert_to_keys(data_keys_or_idx)
        return [self.tokens[val] for val in data_keys]

    @staticmethod
    def load_original_sentences(data_keys):
        # Get relevant sentence numbers
        relevant_sentences = [key[1] for key in data_keys]

        # Connect to database
        database_path = Path(ProjectPaths.tensor_provider, "all_programs.db")
        connection = sqlite3.connect(str(database_path))
        cursor = connection.cursor()

        # Get all sentences with those numbers
        cursor.execute("SELECT program_id, sentence_id, sentence FROM programs WHERE sentence_id IN {}"
                       .format(tuple(relevant_sentences)))

        # Make a dictionary with results
        results_dict = {(val[0], val[1]): val[2] for val in cursor.fetchall()}

        return [results_dict[key] for key in data_keys]

    def load_data_tensors(self, data_keys_or_idx, word_counts=False, char_counts=False,
                          word_embedding=False, word_embedding_success=False,
                          pos_tags=False, char_embedding=False,
                          bow=False, embedding_sum=False, embedding_mean=False,
                          labels=False):
        data_tensors = dict()

        data_keys = self._convert_to_keys(data_keys_or_idx)

        # Get tokens of data-query
        tokens = [self.tokens[val] for val in data_keys]

        # Determine number of words in each sample
        n_words = max([len(text) for text in tokens])

        # Word embeddings
        if word_embedding:
            data_tensors["word_embedding"] = self._get_word_embeddings(tokens=tokens, n_words=n_words)

        # Pos tags
        if pos_tags:
            data_tensors["pos_tags"] = self._get_pos_tags(data_keys=data_keys, n_words=n_words)

        # Character embedding
        if char_embedding:
            data_tensors["char_embedding"] = self._get_character_embedding(data_keys=data_keys, n_words=n_words)

        # BoW representations
        if bow:
            data_tensors["bow"] = self._get_bow_tensors(tokens=tokens)

        # Summed word-embeddings
        if embedding_sum:
            data_tensors["embedding_sum"] = self._get_word_embeddings_sum(tokens=tokens)

        # Meaned word-embeddings
        if embedding_mean:
            data_tensors["embedding_mean"] = self._get_word_embeddings_sum(tokens=tokens, do_mean=True)

        # Data labels
        if labels:
            data_tensors["labels"] = self._get_labels(data_keys=data_keys)

        # Word counts
        if word_counts:
            data_tensors["word_counts"] = np.array([len(text) for text in tokens])

        # Character counts
        if char_counts:
            data_tensors["char_counts"] = np.array([sum([len(word) for word in text])
                                                    for text in tokens])

        return data_tensors

    def get_ranked_predictions(self, predictions, sentence_indices, n_best=10):
        """

        :return: list[str] containing the n_best claims with
                            highest probability under the model
        """
        assert(n_best < len(predictions))
        assert(len(predictions) == len(sentence_indices))

        # Sort predictions
        best_indices = np.argsort(predictions)[-n_best:]
        best_sentences = []
        best_sentences_score = []
        best_test_indices = []
        for e in reversed(range(n_best)):
            best_sentences.append(sentence_indices[best_indices[e]])
            best_sentences_score.append(predictions[best_indices[e]])

        # Return best sentences
        return self.load_original_sentences(best_sentences), best_sentences_score, best_sentences


def reshape_square(a_matrix, pad_value=0, return_pad_mask=False):
    """
    Reshapes any weirds sized numpy-tensor into a square matrix that can be plotted.
    :param np.ndarray a_matrix:
    :param float | int pad_value: Valuee to pad with.
    :param bool return_pad_mask:
    :return:
    """

    def pad_method(x, padding_mode, n_elements):
        if isinstance(padding_mode, str):
            x = np.pad(x, [0, n_elements],
                       mode=padding_mode)
        else:
            x = np.pad(x, [0, n_elements],
                       mode="constant", constant_values=padding_mode)
        return x

    flattened = a_matrix.flatten()
    current_elements = flattened.shape[0]
    sides = int(np.ceil(np.sqrt(current_elements)))
    total_elements = int(sides ** 2)
    pad_elements = total_elements - current_elements
    flattened = pad_method(flattened, pad_value, pad_elements)
    square = flattened.reshape((sides, sides))

    if not return_pad_mask:
        return square
    else:
        mask_square = np.zeros(a_matrix.shape).flatten()
        mask_square = pad_method(mask_square, 1, pad_elements)
        mask_square = mask_square.reshape((sides, sides))  # type: np.ndarray
        return square, mask_square


if __name__ == "__main__":
    sparse = {"bow"}
    plt.close("all")

    # Save tests
    results_dir = Path(ProjectPaths.results, "tensor_provider_tests")
    ensure_folder(results_dir)
    redirect_stdout_to_file(Path(results_dir, "log.txt"))

    # Initialize
    the_tensor_provider = TensorProvider(verbose=True)

    # Get accessible keys
    all_keys = list(sorted(the_tensor_provider.accessible_annotated_keys))

    print("\nTesting tensor provider.")
    the_test_keys = random.sample(all_keys, 20)
    test = the_tensor_provider.load_data_tensors(the_test_keys,
                                                 word_counts=True,
                                                 char_counts=True,
                                                 word_embedding=True,
                                                 word_embedding_success=True,
                                                 pos_tags=True,
                                                 char_embedding=True,
                                                 bow=True,
                                                 embedding_sum=True,
                                                 # embedding_mean=True,
                                                 labels=True)
    test_tokens = the_tensor_provider.load_tokens(the_test_keys)

    print("Original sentences:")
    ori_sentences = the_tensor_provider.load_original_sentences(sorted(the_test_keys))
    for a_key, a_sentence in zip(sorted(the_test_keys), ori_sentences):
        print("\t", a_key, a_sentence)

    print("Shapes:")
    for a_key, a_val in test.items():
        if isinstance(a_val, dict):
            print("\t{} : dict {}".format(a_key, len(a_val)))
        elif isinstance(a_val, np.ndarray):
            print("\t{} : Array {}".format(a_key, a_val.shape))
        else:
            print("\t{} : Unknown type: {}".format(a_key, type(a_val).__name__))

    fig_count = 0
    for a_key, tensor in test.items():
        if not isinstance(tensor, dict):
            plt.figure()

            if len(tensor.shape) == 1:
                plt.imshow(np.expand_dims(tensor, 0), aspect="auto")
                plt.xlabel("Sample")
                plt.yticks([])
                plt.colorbar()

            elif len(tensor.shape) == 2:
                if a_key in sparse:
                    xx, yy = np.where(tensor != 0)
                    plt.scatter(xx, yy)
                else:
                    plt.imshow(tensor.T, aspect="auto")
                plt.xlabel("Sample")
                plt.ylabel("Features")

            else:
                the_rows = cols = np.math.ceil(np.math.sqrt(tensor.shape[0]))
                if the_rows * (cols - 1) == tensor.shape[0]:
                    cols -= 1
                for nr in range(tensor.shape[0]):
                    plt.subplot(the_rows, cols, nr + 1)
                    plt.imshow(tensor[nr, :, :])
                    plt.xlabel(a_key)
                    plt.ylabel("Time")
            plt.suptitle(a_key)

            save_fig(Path(results_dir, "figure_{}".format(fig_count)))
            fig_count += 1
        else:
            print(a_key)
            print(test[a_key])

    n = len(the_tensor_provider._keys)
    assert all([len(a_val) == n for a_val in [
        the_tensor_provider._keys,
        the_tensor_provider.labels,
        the_tensor_provider.tokens,
        the_tensor_provider.pos_tags
    ]]), "Not all resources in TensorProvider has same length."

    ###########
    # Test that all annotated keys have a label and that all non-annotated do not

    print("\nTesting whether all annotated keys have a label and that all non-annotated do not")
    annotated_keys = the_tensor_provider.accessible_annotated_keys
    unannotated_keys = list(set(the_tensor_provider.accessible_keys).difference(set(annotated_keys)))

    annotated_labels = the_tensor_provider.load_labels(annotated_keys, convert_none_to_false=False)
    unannotated_labels = the_tensor_provider.load_labels(unannotated_keys, convert_none_to_false=False)
    assert all([val is None for val in unannotated_labels])
    assert not any([val is None for val in annotated_labels])

    print("Test SUCCESSFUL\n")

    ###########
    # Test difference in data for all samples in program

    distances_tests = [
        dict(word_embedding=True, pos_tags=True, char_embedding=True),
        dict(word_embedding=True, pos_tags=False, char_embedding=False),
        dict(word_embedding=False, pos_tags=True, char_embedding=False),
        dict(word_embedding=False, pos_tags=False, char_embedding=True),
    ]

    # Elements keys
    keys = list(sorted(the_tensor_provider.accessible_annotated_keys))

    # Get program ids and number of programs
    program_ids = np.array(list(zip(*keys))[0])
    unique_programs = set(program_ids)

    # Select test-programs and training-programs
    test_programs = np.random.choice(list(unique_programs),
                                     size=1,
                                     replace=False)

    # Get test-indices
    test_idx = np.sum([program_ids == val for val in test_programs], axis=0)
    test_idx = np.where(test_idx > 0.5)[0]

    # Convert to keys
    test_idx = [keys[val] for val in test_idx]

    for distance_test_nr, settings in enumerate(distances_tests):
        print("Distance test {}".format(distance_test_nr))

        # Get data for keys
        print("\tGetting data")
        # TODO: Perhaps fetch data incrementally like in the comments below, to see if that causes problems.
        # test_idx_copy = list(test_idx)
        # data = []
        # cut_size = int(len(test_idx) / 10)
        # while test_idx_copy:
        #     c_idx, test_idx_copy = test_idx_copy[:cut_size], test_idx_copy[cut_size:]
        #     c_data = the_tensor_provider.load_concat_input_tensors(
        #         data_keys_or_idx=c_idx,
        #         **settings
        #     )
        #     data.append(c_data)
        # data = np.concatenate(data, axis=1)
        data = the_tensor_provider.load_concat_input_tensors(
                data_keys_or_idx=test_idx,
                **settings
            )

        print("\tPreparing")
        # Reshape by stacking all dimensions for each data-point
        # data: [samples, all_dimensions_and_time]
        data = data.reshape((len(test_idx), -1))

        # Broadcast to subtract all rows from all rows
        # data_broad: [samples, 1, all_dimensions_and_time]
        data_broad = data[:, None, :]

        print("\tComputing difference")
        # Compute difference between all rows
        # difference: [samples, samples, all_dimensions_and_time]
        difference = data - data_broad

        print("\tSquaring difference")
        # Square difference
        # difference: [samples, samples, all_dimensions_and_time]
        difference = np.square(difference)

        print("\tComputing distances")
        # Distances
        # difference: [samples, samples]
        distances = np.sqrt(difference.sum(2))

        # Delete this massive object
        del difference

        print("\tPlotting figure")
        # Plot the distance plot
        plt.close("all")
        plt.figure()
        plt.imshow(distances)
        plt.title("Distance between features of all samples in programs {}\n{}".format(test_programs,
                                                                                       settings))
        plt.xlabel("Dimensions: {}".format(the_tensor_provider.input_dimensions(**settings)))
        print("\tSaving figure")
        # Save figure
        save_fig(Path(results_dir, "distances_{}".format(distance_test_nr)))

    print("Done")
    close_stdout_file()
