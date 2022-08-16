import torch
import torch.nn as nn

from transformers import BertTokenizer, BertModel
from typing import List, Tuple, Any, Dict
from fbRecommendation.dl.base.image_text_util import ImageTextUtil
from torchvision.models import (
    resnet50,
    ResNet50_Weights,
    efficientnet_v2_s,
    efficientnet_v2_m,
    EfficientNet_V2_S_Weights,
    EfficientNet_V2_M_Weights
)
from torch.nn.utils.rnn import pad_sequence


class PTImageTextUtil(ImageTextUtil):
    """
    This class is a subclass ImageTextUtil, provide utils functions for tensorflow model
    """

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
            trainable: Whether the returned Word embedding model is trainable

        Returns:
            Any: Embedding model.

        """

        if embedding == 'BERT':
            bert_preprocess_model = BertTokenizer.from_pretrained(pretrain_model)
            bert_model = BertModel.from_pretrained(pretrain_model)
            return bert_model, bert_preprocess_model

        return ImageTextUtil.prepare_embedding_model(
            embedding,
            embedding_dim,
            training_data,
            window,
            min_count,
            pretrain_model,
            trainable
        )

    @staticmethod
    def gensim_to_pytorch_embedding(model: Any):
        """Get a Keras 'Embedding' layer with weights set from Word2Vec model's learned word embeddings.

        Args:
            model: Any
                Gensim word embedding model. This can be Word2vec, FastText or Glove

        Returns:
            `torch.nn.Embedding`
                Embedding layer, to be used as input to deeper network layers.

        """

        weights = torch.FloatTensor(model.wv.vectors)
        embedding = nn.Embedding.from_pretrained(weights)

        for param in embedding.parameters():
            param.requires_grad = False

        return embedding

    @staticmethod
    def prepare_image_base_model(model_name: str, image_shape: Tuple):
        """
        Load an image based model (in tensorflow model format) for transfer learning. This will freeze all the layers
        by default, please call set_base_model_trainable if necessary.

        Args:
            model_name: Based model name
            image_shape: Input shape of the model input.

        Returns:
            model: The model in tf.keras.Model or some others acceptable tensorflow model format.

        """

        if model_name == "RestNet50":
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            # remove final layer
            model = torch.nn.Sequential(*(list(model.children())[:-1]))

        elif model_name == "EfficientNetB0":
            model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
            # remove final layer
            model = torch.nn.Sequential(*(list(model.children())[:-1]))

        elif model_name == "EfficientNetB3":
            model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)
            # remove final layer
            model = torch.nn.Sequential(*(list(model.children())[:-1]))

        for param in model.parameters():
            param.requires_grad = False

        return model

    @staticmethod
    def set_base_model_trainable(model: Any, num_trainable_layer: int):
        """
        Unfreeze the model layers
        Args:
            model: Any pytorch model/sequential layers.
            num_trainable_layer: Number of layers to be freeze/unfreeze. For example, if num_trainable_layer = 50,
                                 the last 50 layers will be set to trainable. To unfreeze all layers, pass in
                                 -1.

        """

        if isinstance(model, nn.Embedding):
            model.weight.requires_grad = True
        else:
            total_layers = len(list(iter(model.parameters())))

            if num_trainable_layer == -1:
                num_trainable_layer = total_layers

            for idx, param in enumerate(model.parameters()):
                if idx > (total_layers - num_trainable_layer):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    @staticmethod
    def prepare_token_with_padding(
            X: Tuple[List[int], List[int], List[int]], embedding: str
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Add padding to the tokens. This is developed for non-transformer based data only. To add padding for transformer based
        model, use batch_encode_text instead.

        Args:
            X: A tuple containing training, validation and testing (in List[int] format), to be padded with trailing
               zero
            embedding: The embedding method. Currently, support Word2Vec only

        Returns:
            A tuple of training, validation and testing dataset

        """
        X_train, X_val, X_test = X

        if embedding == "Word2Vec":
            # add padding
            text_with_padding = [torch.IntTensor(x) for x in X_train]
            text_with_padding.extend([torch.IntTensor(x) for x in X_val])
            text_with_padding.extend([torch.IntTensor(x) for x in X_test])

            text_with_padding = pad_sequence(text_with_padding, True, 0)

            train_end_idx = len(X_train)
            val_end_idx = len(X_train) + len(X_val)

            X_train = text_with_padding[:train_end_idx]
            X_val = text_with_padding[train_end_idx:val_end_idx]
            X_test = text_with_padding[val_end_idx:]

            return X_train, X_val, X_test

    @staticmethod
    def batch_encode_text(
            X: Tuple[List[str], List[str], List[str]], tokenizer: Any
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Encode the text into token ids with padding. This is developed for transformer based data only. To add padding
        for non-transformer based model, use prepare_token_with_padding instead.
        Args:
            X: A tuple containing training, validation and testing (in List[int] format), to be padded with trailing
               zero
            tokenizer: Tokeniser that matches the embedding model

        Returns:
            Tuple[Dict, Dict, Dict]:
                A tuple of dictionary containing the encoded and padded data of training, validation and testing data.

        """

        X_train, X_val, X_test = X

        train_end_idx = len(X_train)
        val_end_idx = len(X_train) + len(X_val)

        text = X_train
        text.extend(X_val)
        text.extend(X_test)

        encoded_text = tokenizer.batch_encode_plus(
            text,
            padding="max_length",
            truncation=True
        )

        encoded_text = {key: torch.LongTensor(value) for key, value in encoded_text.items()}

        X_train = {key: value[:train_end_idx] for key, value in encoded_text.items()}
        X_val = {key: value[train_end_idx:val_end_idx] for key, value in encoded_text.items()}
        X_test = {key: value[val_end_idx:] for key, value in encoded_text.items()}

        return X_train, X_val, X_test
