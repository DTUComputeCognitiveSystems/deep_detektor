import csv
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
        # Labels of original data

        if verbose:
            print("Loading labels.")
        self.labels = dict()
        self.keys = []
        database_path = Path(ProjectPaths.tensor_provider, "annotated_programs.db")
        connection = sqlite3.connect(str(database_path))
        cursor = connection.cursor()
        rows = cursor.execute("SELECT program_id, sentence_id, claim_flag FROM programs").fetchall()
        for row in rows:
            key = (row[0], row[1])
            self.keys.append(key)
            self.labels[key] = bool(row[2])
        cursor.close()
        connection.close()

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
            data_keys_or_idxs = [self.keys[val] for val in data_keys_or_idxs]

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

    def load_labels(self, data_keys_or_idx):
        data = self.load_data_tensors(data_keys_or_idx=data_keys_or_idx,
                                      labels=True)
        return data["labels"]

    def load_tokens(self, data_keys_or_idx):
        data_keys = self._convert_to_keys(data_keys_or_idx)
        return [self.tokens[val] for val in data_keys]

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


def reshape_square(a_matrix, pad_value=0, return_pad_mask=False):
    """
    Reshapes any weirds sized numpy-tensor into a square matrix that can be plotted.
    :param np.ndarray a_matrix:
    :param float | int pad_value: Valuee to pad with.
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

    tensor_provider = TensorProvider(verbose=True)
    print("\nTesting tensor provider.")
    test_nrs = random.sample(range(len(tensor_provider.keys)), 20)
    data_keys = tensor_provider._convert_to_keys(test_nrs)
    test = tensor_provider.load_data_tensors(test_nrs,
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
    test_tokens = tensor_provider.load_tokens(test_nrs)

    print("Shapes:")
    for key, val in test.items():
        if isinstance(val, dict):
            print("\t{} : dict {}".format(key, len(val)))
        elif isinstance(val, np.ndarray):
            print("\t{} : Array {}".format(key, val.shape))
        else:
            print("\t{} : Unknown type: {}".format(key, type(val).__name__))

    for key, tensor in test.items():
        if not isinstance(tensor, dict):
            plt.figure()

            if len(tensor.shape) == 1:
                plt.imshow(np.expand_dims(tensor, 0), aspect="auto")
                plt.xlabel("Sample")
                plt.yticks([])
                plt.colorbar()

            elif len(tensor.shape) == 2:
                if key in sparse:
                    x, y = np.where(tensor != 0)
                    plt.scatter(x, y)
                else:
                    plt.imshow(tensor.T, aspect="auto")
                plt.xlabel("Sample")
                plt.ylabel("Features")

            else:
                rows = cols = np.math.ceil(np.math.sqrt(tensor.shape[0]))
                if rows * (cols - 1) == tensor.shape[0]:
                    cols -= 1
                for nr in range(tensor.shape[0]):
                    plt.subplot(rows, cols, nr+1)
                    plt.imshow(tensor[nr, :, :])
                    plt.xlabel(key)
                    plt.ylabel("Time")
            plt.suptitle(key)
        else:
            print(key)
            print(test[key])


