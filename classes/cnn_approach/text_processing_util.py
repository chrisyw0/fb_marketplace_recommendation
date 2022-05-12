import tensorflow as tf
import os
import re
from typing import List, Tuple, Any
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from gensim.models import Word2Vec


class TextUtil:
    @staticmethod
    def tokenise(product_sentences: List[List[str]],
                 embedding: str,
                 with_symbol: bool = False) -> Tuple[List[str], List[List[str]], int]:

        """
        Tokenise the input sentence into tokens.
        Args:
            product_sentences: The products' description sentences.
            embedding: Type of embedding.
            with_symbol: Keep symbols in the list of tokens.

        Returns:
            Tuple[List[str], List[List[str]], int]:
                A list of tokens of the input product
                Split input into sentences, and each sentence is a list of tokens
                Maximum number of tokens of the input products' sentences

        """

        if embedding == 'Word2Vec':
            nlp = English()
            tokenizer = Tokenizer(nlp.vocab)

            clean_product_tokens = []  # store a list tokens for each product
            w2v_training_text = []  # store a list of sentences for all products

            num_max_tokens = 0

            # for sentences in each product
            for sentences in product_sentences:
                this_product_tokens = []

                # for each sentence in the product
                for sentence in sentences:

                    # trim the leading and trailing space, convert to lower case
                    # keep only a-z, and finally tokenise it
                    text = sentence.strip()
                    text = text.lower()

                    if not with_symbol:
                        text = re.sub(r'[^\w\s]', '', text)

                    tokens = tokenizer(text)
                    tokens = [token.text for token in list(tokens)]
                    tokens = list(filter(lambda x: len(x.strip()) > 0, tokens))

                    if len(tokens) > 0:
                        this_product_tokens.extend(tokens)
                        w2v_training_text.append(tokens)

                clean_product_tokens.append(this_product_tokens)
                num_max_tokens = max(num_max_tokens, len(this_product_tokens))

            return clean_product_tokens, w2v_training_text, num_max_tokens

    @staticmethod
    def prepare_embedding_model(embedding: str,
                                embedding_dim: int,
                                training_data: List[List[str]] = None,
                                window: int = 2,
                                min_count: int = 1,
                                pretrain_model: str = None) -> Any:

        """
        Create a word embedding model.
        Args:
            embedding: Type of embedding to be used for the model
            embedding_dim: Dimension of the embedding output
            training_data: Training data to train the model.
            window: The window size to be used to train the model.
            min_count: The minimum number token to be found in training dataset and to be included in the embedding model.
            pretrain_model: The pre-train model to be used.

        Returns:
            Any: Embedding model.

        """

        if embedding == 'Word2Vec':
            os.makedirs(f"./model/{embedding}/", exist_ok=True)

            # this will train the Word2Vec model with given sentence
            model = Word2Vec(sentences=training_data,
                             vector_size=embedding_dim,
                             window=window,
                             min_count=min_count)

            model.save(f"./model/{embedding}/{embedding}.model")

            return model

    @staticmethod
    def get_token_index(product_tokens: List[List[str]],
                        embedding: str,
                        model: Any) -> List[List[int]]:

        """
        Get token index for each token in the embedding model.

        Args:
            product_tokens: List of product tokens.
            embedding: Type of embedding.
            model: Embedding model.

        Returns:
            A list of token index for each input product's tokens.

        """

        # convert token to index in the model, in the embedding model, we can expect
        # each row record in the weight matrix represent the vector of a word. So the
        # model can convert the word from the index to the embedding from this index.
        if embedding == "Word2Vec":
            tokens_idx = []

            for tokens in product_tokens:
                this_token_idx = []

                for token in tokens:
                    this_token_idx.append(model.wv.key_to_index[token])

                tokens_idx.append(this_token_idx)

            return tokens_idx

    @staticmethod
    def gensim_to_keras_embedding(model: Any,
                                  train_embeddings: bool = False,
                                  input_shape : Tuple = None):
        """Get a Keras 'Embedding' layer with weights set from Word2Vec model's learned word embeddings.

        Parameters
        ----------
        model: Any
            Gensim word embedding model. This can be Word2vec, FastText or Glove
        train_embeddings : bool
            If False, the returned weights are frozen and stopped from being updated.
            If True, the weights can / will be further updated in Keras.
        input_shape: Tuple
            Input shape to the output layer

        Returns
        -------
        `keras.layers.Embedding`
            Embedding layer, to be used as input to deeper network layers.

        """

        # this is copied from gensim github with additional input shape input as param.

        keyed_vectors = model.wv  # structure holding the result of training
        weights = keyed_vectors.vectors  # vectors themselves, a 2D numpy array
        index_to_key = keyed_vectors.index_to_key  # which row in `weights` corresponds to which word?

        layer = tf.keras.layers.Embedding(
            input_dim=weights.shape[0],
            output_dim=weights.shape[1],
            weights=[weights],
            trainable=train_embeddings,
            input_shape=input_shape
        )

        return layer
