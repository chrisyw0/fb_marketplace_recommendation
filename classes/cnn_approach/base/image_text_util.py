import os
import re
import emoji

from typing import List, Tuple, Any
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import remove_stopwords


class ImageTextUtil:
    """
    This class provides some common util methods for image and text processing
    """

    @staticmethod
    def clean_text(text: str, remove_stop_words: bool = False) -> str:
        """
        Remove emoji, tab, new line and spaces in the text
        Args:
            text: The text to be cleaned
            remove_stop_words: Whether to remove stop words from the text

        Returns:
            str:
                Cleaned text

        """
        result_text = emoji.replace_emoji(text, replace='')
        result_text = re.sub('[\n\t\r|]', '', result_text)
        result_text = re.sub(' +', ' ', result_text)

        if remove_stop_words:
            result_text = remove_stopwords(result_text)

        result_text = result_text.strip()

        return result_text

    @staticmethod
    def tokenise(product_sentences: List[str],
                 embedding: str,
                 with_symbol: bool = False,
                 pre_trained_model: str = None) -> Tuple[List[str], Any, int]:

        """
        Tokenise the input sentence into tokens.
        Args:
            product_sentences: The products' description sentences.
            embedding: Type of embedding.
            with_symbol: Keep symbols in the list of tokens.
            pre_trained_model: The name of pretrained model

        Returns:
            Tuple[List[str], List[List[str]], int]:
                A list of tokens of the input product
                Any extra data
                Maximum number of tokens of the input products' sentences

        """

        if embedding == 'Word2Vec':
            nlp = English()
            tokenizer = Tokenizer(nlp.vocab)

            clean_product_tokens = []  # store a list tokens for each product
            w2v_training_text = []  # store a list of sentences for all products

            num_max_tokens = 0

            product_sentences = [sentence.split('.') for sentence in product_sentences]

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
                                pretrain_model: str = None,
                                trainable: bool = True) -> Any:

        """
        Create a word embedding model.
        Args:
            embedding: Type of embedding to be used for the model
            embedding_dim: Dimension of the embedding output
            training_data: Training data to train the model.
            window: The window size to be used to train the model.
            min_count: The minimum number token to be found in training dataset and to be included in the embedding model.
            pretrain_model: The pre-train model to be used.
            trainable: Whether the model is trainable

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
    def load_embedding_model(embedding: str) -> Any:
        """
        Load the gensim model from saved path
        Args:
            embedding: embedding method name

        Returns:
            Any: Embedding model.
        """
        if embedding == 'Word2Vec':
            model = Word2Vec.load(f"./model/{embedding}/{embedding}.model")
            return model

        raise ValueError(f"model not found for embedding method {embedding}")

    @staticmethod
    def get_token_index(product_tokens: List[str],
                        embedding: str,
                        model: Any,
                        extra_data: Any = None) -> List[List[int]]:

        """
        Get token index for each token in the embedding model.

        Args:
            product_tokens: List of product tokens.
            embedding: Type of embedding.
            model: Embedding model.
            extra_data: Extra data that help build the model.

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
    def prepare_image_base_model(model_name: str, trainable: bool = False):
        """
        Abstract method to return an image based model using as part of transfer learning.
        Args:
            model_name: The based model name
            trainable: Whether the model is trainable

        Returns:
            The based model
        """
        pass

