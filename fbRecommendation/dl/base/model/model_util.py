import os
from typing import List, Any, Tuple
from gensim.models import Word2Vec


class ModelUtil:
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
    def prepare_image_base_model(model_name: str, image_shape: Tuple, trainable: bool = False):
        """
        Abstract method to return an image based model using as part of transfer learning.
        Args:
            model_name: The based model name
            trainable: Whether the model is trainable

        Returns:
            The based model
        """
        pass
