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
    def __init__(self, tf_graph, tf_session, verbose=False, end_of_word_char="$", fill=np.nan):
        # Make graph and session
        self.fill = fill
        self._tf_graph = tf_graph  # tf.Graph() if tf_graph is None else tf_graph
        self._sess = tf_session  # tf.Session() if tf_session is None else tf_session
        self._sess.run(tf.global_variables_initializer())

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

        self.speller_dimensions = speller_results['cells']
        self.recurrent_speller = RecurrentSpeller(n_inputs=len(self.char_embedding),
                                                  n_outputs=len(self.char_embedding),
                                                  n_encoding_cells=self.speller_dimensions,
                                                  n_decoding_cells=self.speller_dimensions,
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
                self.labels[key] = row[6]

        ###################
        # Word embeddings

        if verbose:
            print("Loading Word-Embeddings.")
        self.word_embeddings = dict()
        with embeddings_file.open("r") as file:
            csv_reader = csv.reader(file, delimiter=",")
            for row in csv_reader:
                self.word_embeddings[row[0]] = np.array(eval(row[1]))
        self.word_embedding_length = len(self.word_embeddings[list(self.word_embeddings.keys())[0]])

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
        self._pos_embedding = np.eye(len(self.pos_vocabulary), len(self.pos_vocabulary))

        ###################
        # BOW-settings

        # Set an transformer for BOW (ex. stemmer)
        stemmer = nltk.stem.SnowballStemmer('danish')
        self.bow_transformer = lambda x: stemmer.stem(x.lower())  # type: Callable

        # The set of known words
        if self.bow_transformer is None:
            self.bow_vocabulary = self.vocabulary
        else:
            bow_vocabulary = {self.bow_transformer(word) for word in self.vocabulary.keys()}
            self.bow_vocabulary = {word: idx for idx, word in enumerate(sorted(list(bow_vocabulary)))}

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
        out_array = np.full((len(tokens), n_words, self.word_embedding_length + 1), self.fill)

        # Default vector (for unknown words)
        default_vector = np.array([0] * self.word_embedding_length + [1])

        # Insert word-vectors
        for text_nr, text in enumerate(tokens):
            for word_nr, word in enumerate(text):
                c_word = self._get_known_word(word)
                if c_word is not None:
                    c_embedding = self.word_embeddings[c_word]
                    out_array[text_nr, word_nr, :-1] = c_embedding
                else:
                    out_array[text_nr, word_nr, :] = default_vector

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
        out_array = np.full((len(data_keys), n_words, self.speller_dimensions), self.fill)

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

    def load_data_tensors(self, data_keys_or_idxs):
        data_tensors = dict()

        if isinstance(data_keys_or_idxs[0], int):
            data_keys_or_idxs = [self.keys[val] for val in data_keys_or_idxs]

        # Get tokens of data-query
        tokens = [self.tokens[val] for val in data_keys_or_idxs]

        # Determine number of words in each sample
        n_words = max([len(val) for val in tokens])

        # Word embeddings
        data_tensors["word_embedding"] = self._get_word_embeddings(tokens=tokens, n_words=n_words)

        # Pos tags
        data_tensors["pos_tags"] = self._get_pos_tags(data_keys=data_keys_or_idxs, n_words=n_words)

        # Character embedding
        data_tensors["char_embedding"] = self._get_character_embedding(data_keys=data_keys_or_idxs, n_words=n_words)

        # BoW representations
        data_tensors["bow"] = self._get_bow_tensors(tokens=tokens)

        return data_tensors


if __name__ == "__main__":
    tensor_provider = TensorProvider(tf_graph=tf.Graph(), tf_session=tf.Session(), verbose=True)
    print("\nTesting tensor provider.")
    test = tensor_provider.load_data_tensors([0, 3, 4, 6])
