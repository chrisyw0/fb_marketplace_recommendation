import torch
import torch.nn as nn

from typing import Tuple, Union, Dict
from torchvision import transforms
from fbRecommendation.dl.pytorch.model.pt_model_util import PTModelUtil


class PTImageModel(nn.Module):
    """
    This is the image model in nn.Module format containing image layers (transformation  + based model +
    image sequential layers) and finally a prediction layer.

    The model override the forward method of the nn.Module which gives instructions how to process the input data
    and give prediction from it.

    It accepts image input format in torch.Tensor, which should be a 4d tensor
    [batch_size, channel, height, width]
    """
    def __init__(
            self,
            num_class: int,
            image_base_model: nn.Module,
            image_shape: Tuple[int, int, int],
            dropout_conv: float,
            dropout_pred: float,
            base_model_output_dim: int
    ):
        super(PTImageModel, self).__init__()
        self.transforms = nn.Sequential(
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(72),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )

        self.image_base_model = PTModelUtil.prepare_image_base_model(
            image_base_model,
            image_shape
        )

        self.sequential_layer = nn.Sequential(
            nn.Dropout(dropout_conv),
            nn.Linear(base_model_output_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(dropout_pred)
        )

        self.prediction_layer = nn.Linear(256, num_class)

    def forward(self, image: torch.Tensor):
        x = self.transforms(image)
        x = self.image_base_model(x)
        x = x.squeeze()
        if x.dim() == 1:
            # if only one data in a batch, it adds back the dimension
            x = x.unsqueeze(0)
        x = self.sequential_layer(x)
        x = self.prediction_layer(x)
        return x


class PTTextModel(nn.Module):
    """
    This is the text model in nn.Module format containing text layers (embedding and text sequential layers)
    and finally a prediction layer. The embedding layer should be non-transformer based model
    i.e. Word2Vec, Fasttext or Glove

    The model override the forward method of the nn.Module which gives instructions how to process the input data
    and give prediction from it.

    It accepts input format in torch.Tensor [batch_size, max_token_number].
    """
    def __init__(
            self,
            num_class,
            embedding_dim,
            embedding_layer,
            dropout_conv,
            dropout_pred
    ):
        super(PTTextModel, self).__init__()
        self.embedding_layer = embedding_layer

        self.sequential_layer = nn.Sequential(
            nn.Conv1d(embedding_dim, 48, 3),
            nn.ReLU(),
            nn.AvgPool1d(2),
            nn.Conv1d(48, 24, 3),
            nn.ReLU(),
            nn.AvgPool1d(2),
            nn.Flatten(),
            nn.Dropout(dropout_conv),
            nn.Linear(8712, 256),
            nn.ReLU(),
            nn.Dropout(dropout_pred)
        )

        self.prediction_layer = nn.Linear(256, num_class)

    def forward(self, text):
        x = self.embedding_layer(text)
        x = torch.transpose(x, 1, 2)
        x = self.sequential_layer(x)
        x = self.prediction_layer(x)
        return x


class PTTextTransformerModel(nn.Module):
    """
    This is the text model in nn.Module format containing text layers (embedding and text sequential layers)
    and finally a prediction layer. The embedding layer should be transformer based model,
    i.e. BERT, Roberta or Longformer

    The model override the forward method of the nn.Module which gives instructions how to process the input data
    and give prediction from it.

    It accepts the format in dictionary format, containing token id, attention mask
    and some other values required from model input, which can be encoded by PTImageTextUtil.batch_encode_text.
    """
    def __init__(
            self,
            num_class: int,
            embedding_dim: int,
            embedding_layer: nn.Module,
            dropout_pred: float
    ):
        super(PTTextTransformerModel, self).__init__()

        self.embedding_layer = embedding_layer

        self.sequential_layer = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_pred)
        )

        self.prediction_layer = nn.Linear(256, num_class)

    def forward(self, text: Dict[str, torch.Tensor]):
        x = self.embedding_layer(**text)["pooler_output"]
        x = self.sequential_layer(x)
        x = self.prediction_layer(x)
        return x


class PTImageTextModel(nn.Module):
    """
    This is the combine model in nn.Module format containing image layers (transformation  + based model +
    image sequential layers), text layers (embedding + text sequential layers) and finally a prediction layer.
    The image and text layers have been trained previously with the same data in text and image only model. The model
    can benefit from both pre-trained layers and hopefully give better performance than having single image or text
    only model.

    The model override the forward method of the nn.Module which gives instructions how to process the input data
    and give prediction from it.

    Similar to image only model, it accepts image input format in torch.Tensor, which should be a 4d tensor
    [batch_size, channel, height, width]

    This model handles both transformer based and non-transformer based text embedding layer. Please note that for
    non-transformer based embedding layer, it accepts input format in torch.Tensor [batch_size, max_token_number]
    while for transformer based model, it accepts the format in dictionary format, containing token id, attention mask
    and some other values required from model input, which can be encoded by PTImageTextUtil.batch_encode_text
    """
    def __init__(
            self,
            num_class: int,
            image_base_model: nn.Module,
            image_seq_layers: nn.Module,
            text_embedding_layer: nn.Module,
            text_seq_layers: nn.Module,
            is_transformer_based_text_model: bool
    ):
        super(PTImageTextModel, self).__init__()
        self.transforms = nn.Sequential(
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(72),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )

        self.image_seq_layers = image_seq_layers
        self.image_base_model = image_base_model

        self.text_embedding_layer = text_embedding_layer
        self.text_seq_layers = text_seq_layers

        self.is_transformer_based_text_model = is_transformer_based_text_model

        self.prediction_layer = nn.Linear(512, num_class)

    def forward(self, image: torch.Tensor, text: Union[torch.Tensor, Dict[str, torch.Tensor]]):
        x_img = self.transforms(image)
        x_img = self.image_base_model(x_img)
        x_img = x_img.squeeze()
        if x_img.dim() == 1:
            # if only one data in a batch, it add back the dimension
            x_img = x_img.unsqueeze(0)
        x_img = self.image_seq_layers(x_img)

        if self.is_transformer_based_text_model:
            x_text = self.text_embedding_layer(**text)["pooler_output"]
        else:
            x_text = self.text_embedding_layer(text)
            x_text = torch.transpose(x_text, 1, 2)

        x_text = self.text_seq_layers(x_text)

        x = torch.cat((x_img, x_text), dim=1)
        x = self.prediction_layer(x)

        return x

