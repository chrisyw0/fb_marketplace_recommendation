import torch
import torch.nn as nn

from typing import Any, List, Tuple, Optional
from transformers import BertTokenizer, BertModel
from torchvision.models import (
    resnet50,
    ResNet50_Weights,
    efficientnet_v2_s,
    efficientnet_v2_m,
    EfficientNet_V2_S_Weights,
    EfficientNet_V2_M_Weights
)
from fbRecommendation.dl.base.model.model_util import ModelUtil


class PTModelUtil(ModelUtil):

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
    def prepare_image_base_model(model_name: str, image_shape: Tuple, trainable: bool = False):
        """
        Load an image based model (in tensorflow model format) for transfer learning. This will freeze all the layers
        by default, please call set_base_model_trainable if necessary.

        Args:
            model_name: Based model name
            image_shape: Input shape of the model input.
            trainable: Whether the base model is trainable. Default to False

        Returns:
            model: The model in tf.keras.Model or some others acceptable tensorflow model format.

        """

        model: Optional[nn.Module] = None

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

        if not model:
            raise ValueError(f"Base model {model_name} not found")

        for param in model.parameters():
            param.requires_grad = trainable

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

        return ModelUtil.prepare_embedding_model(
            embedding,
            embedding_dim,
            training_data,
            window,
            min_count,
            pretrain_model,
            trainable
        )
