import csv
import json
from collections import Counter
from typing import Callable

import nltk
import numpy as np
import tensorflow as tf

from models.speller.recurrent_speller import RecurrentSpeller
from project_paths import embeddings_file, pos_tags_file, speller_results_file, speller_char_vocab_file, \
    speller_translator_file, speller_encoder_checkpoint_file, data_matrix_path


class TensorProvider:
    def __init__(self, verbose=False, end_of_word_char="$", fill=np.nan):
        self.fill = fill

        # Make graph and session
        self._tf_graph = tf.Graph()
        self._sess = tf.Session(graph=self._tf_graph)

        ###################
        # Character embedding (auto-encoder)

        if verbose:
            print("Loading character embedding.")
        with speller_char_vocab_file.open("r") as file:
            self.char_embedding = json.load(file)
        with speller_results_file.open("r") as file:
            speller_results = json.load(file)
        with speller_translator_file.open("r") as file:
            self.string_translator = {int(val[0]): val[1] for val in json.load(file).items()}

        with self._tf_graph.as_default():
            self.char_embedding_size = speller_results['cells']
            self.recurrent_speller = RecurrentSpeller(n_inputs=len(self.char_embedding),
                                                      n_outputs=len(self.char_embedding),
                                                      n_encoding_cells=self.char_embedding_size,
                                                      n_decoding_cells=self.char_embedding_size,
                                                      character_embedding=self.char_embedding,
                                                      end_of_word_char=end_of_word_char)
            self.recurrent_speller.load_encoder(sess=self._sess, file_path=speller_encoder_checkpoint_file)

        ###################
        # Labels of original data

        if verbose:
            print("Loading labels.")
        self.labels = dict()
        self.keys = []
        with data_matrix_path.open("r", encoding="utf-8") as file:
            csv_reader = csv.reader(file, delimiter=",")

            # Skip header
            next(csv_reader)

            # Load file
            for row in csv_reader:
                key = (int(row[2]), int(row[3]))
                self.keys.append(key)
                self.labels[key] = bool(row[6])

        ###################
        # Word embeddings

        if verbose:
            print("Loading Word-Embeddings.")
        self.word_embeddings = dict()
        with embeddings_file.open("r") as file:
            csv_reader = csv.reader(file, delimiter=",")
            for row in csv_reader:
                self.word_embeddings[row[0]] = np.array(eval(row[1]))

        # Word embedding length (+1 due to flag for unknown vectors)
        self.word_embedding_size = len(self.word_embeddings[list(self.word_embeddings.keys())[0]]) + 3

        ###################
        # Tokenized texts and POS-tags

        if verbose:
            print("Loading POS-taggings and tokenized elements.")
        self.pos_tags = dict()
        self.tokens = dict()
        self.pos_vocabulary = set()
        self.vocabulary = set()
        with pos_tags_file.open("r") as file:
            csv_reader = csv.reader(file, delimiter=",")
            for row in csv_reader:
                key = (int(eval(row[0])), int(eval(row[1])))
                pos_tags = eval(row[2])
                tokens = [val.decode() for val in eval(row[3])]

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

        # Set an transformer for BOW (ex. stemmer)
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

    def _get_known_word(self, word: str):
        operations = [
            lambda x: x,
            lambda x: x.translate(self.string_translator),
            lambda x: self.bow_transformer(x),
            lambda x: self.bow_transformer(x.translate(self.string_translator))
        ]

        for operation in operations:
            c_word = operation(word)
            if c_word in self.word_embeddings:
                return c_word

        # Word not known
        return None

    def _get_word_embeddings(self, tokens, n_words=None):

        # Determine longest sentence (in word-count)
        if n_words is None:
            n_words = max([len(val) for val in tokens])

        # Initialize array
        out_array = np.full((len(tokens), n_words, self.word_embedding_size), self.fill)

        # Default vector (for unknown words)
        unknown_vector = np.array([0] * self.word_embedding_size)
        unknown_vector[-1] = 1

        # Insert word-vectors
        for text_nr, text in enumerate(tokens):
            for word_nr, word in enumerate(text):
                c_word = self._get_known_word(word)
                if c_word is not None:
                    c_embedding = self.word_embeddings[c_word]
                    out_array[text_nr, word_nr, :-1] = c_embedding
                else:
                    out_array[text_nr, word_nr, :] = unknown_vector

        # Return
        return out_array

    def _get_pos_tags(self, data_keys, n_words):
        # Initialize array
        out_array = np.full((len(data_keys), n_words, len(self.pos_vocabulary)), self.fill)

        # Insert pos-tags
        for key_nr, key in enumerate(data_keys):
            c_pos_tags = np.array([self.pos_vocabulary[val] for val in self.pos_tags[key]
                                   if val != "PUNCT"])

            out_array[key_nr, :c_pos_tags.shape[0], :] = self._pos_embedding[:, c_pos_tags].T

        return out_array

    def _get_character_embedding(self, data_keys, n_words):
        # Initialize array
        out_array = np.full((len(data_keys), n_words, self.char_embedding_size), self.fill)

        # Go through elements
        for key_nr, key in enumerate(data_keys):
            c_words = self.tokens[key]  # type: list
            c_words = [val.translate(self.string_translator) for val in c_words]
            c_words = ["".join([char for char in word if word in self.char_embedding]) for word in c_words]

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

    def input_dimensions(self, word_embedding=True, pos_tags=True, char_embedding=True, bow=False):
        # Assert working with either global features or sequencial features
        assert not any([bow]) or not any([word_embedding, pos_tags, char_embedding])
        d = 0

        if word_embedding:
            d += self.word_embedding_size

        if pos_tags:
            d += self.pos_embedding_size

        if char_embedding:
            d += self.char_embedding_size

        if bow:
            d += len(self.bow_vocabulary)

        return d

    def load_data_tensors(self, data_keys_or_idx,
                          word_embedding=True, pos_tags=True, char_embedding=True, bow=True,
                          labels=True):
        data_tensors = dict()

        data_keys = self._convert_to_keys(data_keys_or_idx)

        # Get tokens of data-query
        tokens = [self.tokens[val] for val in data_keys]

        # Determine number of words in each sample
        n_words = max([len(val) for val in tokens])

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

        # Data labels
        if labels:
            data_tensors["labels"] = self._get_labels(data_keys=data_keys)

        return data_tensors


if __name__ == "__main__":
    tensor_provider = TensorProvider(verbose=True)
    print("\nTesting tensor provider.")
    test = tensor_provider.load_data_tensors([0, 3, 4, 6])
