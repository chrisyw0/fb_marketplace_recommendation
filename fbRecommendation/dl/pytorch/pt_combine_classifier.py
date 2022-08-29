import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import copy

from typing import Tuple, List, Any, Optional, Union, Dict
from dataclasses import field
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from fbRecommendation.dl.pytorch.utils.pt_image_text_util import PTImageTextUtil
from .pt_base_classifier import (
    PTBaseClassifier,
    train_and_validate_model,
    evaluate_model,
    predict_model,
    prepare_optimizer_and_scheduler
)

from fbRecommendation.dl.pytorch.utils.pt_dataset_generator import PTImageTextDataset
from fbRecommendation.dataset.prepare_dataset import DatasetHelper


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


class PTImageTextClassifier(PTBaseClassifier):
    """
    A deep learning model predicting product category from its image, name and description.

    Args:
        df_image (pd.DataFrame): Image dataframe
        df_product (pd.DataFrame): Product dataframe

        image_base_model_name (str, optional): Name of the image pre-trained model. Defaults to "EfficientNetB3"
        embedding (str, Optional): The type of embedding model. Defaults to "BERT".

        embedding_pretrain_model (str, Optional): Whether to use a pretrain model to encode the text. This is used to
                                                  get the tokenizer for transformer based model.
                                                  Defaults to "bert-base-cased", which is the pre-trained model of BERT.
        embedding_dim (int, Optional): The dimension of the embedding model. Defaults to 768.
        image_path: (str, optional): Path to store the original image. Defaults to "./data/images/"
        transformed_image_path (str, optional): Path to cache the transformed image. This is improving when training,
                                                validating and testing the model as we don't need to transform and resize
                                                images when they are loaded into memory. Defaults to "./data/adjusted_img/"

        image_shape (Tuple[int, int, int], Optional): Size of the image inputting to the model.
                                                      If image channel = 'RGB', the value will be
                                                      (width, height, 3) i.e. 3 channels
                                                      Defaults to (300, 300, 3)

        epoch (float, optional): Epoch of the model. Defaults to 3.
        learning_rate (float, optional): Learning rate of the model. Defaults to 1e-3.
        batch_size (int, optional): Batch size of the model. Defaults to 16.

        metrics (List[str], optional):  list of metrics using for model evaluation. Defaults to ["accuracy"].

        image_base_model (nn.Module): The trained image based model.
        image_seq_layers (nn.Module): The trained image sequential layers (without preprocessing and prediction layer).
        text_embedding_layer (nn.Module): The trained embedding model.
        text_seq_layers (nn.Module): The trained text sequential layers (without preprocessing and prediction layer).

    """
    df_product: pd.DataFrame
    df_image: pd.DataFrame

    image_base_model_name: str = "EfficientNetB3"
    embedding: str = "BERT"
    embedding_dim: int = 768

    image_path: str = "./data/images/"
    transformed_image_path: str = "./data/adjusted_img/"

    embedding_pretrain_model: str = "bert-base-cased"

    image_shape: Tuple[int, int, int] = (300, 300, 3)
    transformed_image_path += str(image_shape[0]) + "/"

    epoch: int = 3
    learning_rate: float = 1e-3
    batch_size: int = 16

    metrics: List[str] = field(default_factory=lambda: ["accuracy"])

    def __init__(
            self,
            df_image: pd.DataFrame,
            df_product: pd.DataFrame,
            image_base_model: Any,
            image_seq_layers: Any,
            text_embedding_layer: Optional[Any],
            text_seq_layers: Any,
            is_transformer_based_text_model
    ):

        super().__init__(df_image, df_product)

        self.image_seq_layers = copy.deepcopy(image_seq_layers)
        self.text_seq_layers = copy.deepcopy(text_seq_layers)
        self.image_base_model = copy.deepcopy(image_base_model)
        self.text_embedding_layer = copy.deepcopy(text_embedding_layer)
        self.is_transformer_based_text_model = is_transformer_based_text_model

        print("Init with previous trained image and text layers")

    def _get_model_name(self):
        return f"pt_image_text_model_{self.image_base_model_name}_{self.embedding}"

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare training, validation and testing data for the model. It includes building the word tokeniser and
        embedding model, splitting the dataset and getting essential elements for later stages.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Training, validation and testing dataframe

        """

        # merge the image and product dataset
        generator = DatasetHelper(self.df_product, self.df_image)
        df_image_data, self.classes = generator.generate_image_product_dataset()

        self.num_class = len(self.classes)

        if not self.is_transformer_based_text_model:
            # split by . to make a list of sentences for each product
            product_sentences = df_image_data['product_name_description'].to_list()

            # whether to keep non word chars
            with_symbol = False if self.embedding == 'Word2Vec' else True

            print("Start tokenising the product name and description")

            # tokenise the sentences, and get training data for the embedding model
            # and get the maximum length in the product descriptions
            product_tokens, training_data, self.num_max_tokens = PTImageTextUtil.tokenise(
                product_sentences,
                self.embedding,
                with_symbol
            )

            print("Getting index from the embedding model")

            # restore a gensim model for getting token index.
            embedding_model = PTImageTextUtil.load_embedding_model(self.embedding)

            # convert token into token index in embedding model weight matrix
            df_image_data['tokens_index'] = PTImageTextUtil.get_token_index(
                product_tokens,
                self.embedding,
                embedding_model
            )

            self.input_shape = (
                (self.image_shape[2], self.image_shape[1], self.image_shape[0]),
                (self.num_max_tokens,)
            )
            self.input_dtypes = [torch.float, torch.int]

            print("Prepare training, validation and testing data")

            # split dataset
            df_train, df_val, df_test = generator.split_dataset(df_image_data)

            X_train = df_train['tokens_index'].to_list()
            X_val = df_val['tokens_index'].to_list()
            X_test = df_test['tokens_index'].to_list()

            X_train, X_val, X_test = PTImageTextUtil.prepare_token_with_padding(
                (X_train, X_val, X_test),
                self.embedding
            )

        else:
            df_image_data['product_name_description'] = df_image_data["product_name_description"].apply(
                PTImageTextUtil.clean_text)

            _, tokenizer = PTImageTextUtil.prepare_embedding_model(
                embedding=self.embedding,
                embedding_dim=self.embedding_dim,
                pretrain_model=self.embedding_pretrain_model,
                trainable=False
            )

            # split dataset
            df_train, df_val, df_test = generator.split_dataset(df_image_data)

            X_train = df_train['product_name_description'].to_list()
            X_val = df_val['product_name_description'].to_list()
            X_test = df_test['product_name_description'].to_list()

            X_train, X_val, X_test = PTImageTextUtil.batch_encode_text((X_train, X_val, X_test), tokenizer)

            input_img = np.random.rand(1, self.image_shape[2], self.image_shape[1], self.image_shape[0])
            input_img = torch.from_numpy(input_img)
            input_img = input_img.float()

            self.input_data = [
                input_img,
                {key: torch.unsqueeze(value[0], dim=0) for key, value in X_train.items()}
            ]

        print("Prepare training, validation and testing data")

        image_train, image_val, image_test = generator.get_image_ids(df_train, df_val, df_test)
        y_train, y_val, y_test = generator.get_product_categories(df_train, df_val, df_test)

        train_ds = PTImageTextDataset(
            images=image_train,
            tokens=X_train,
            image_root_path=self.image_path,
            image_shape=self.image_shape,
            transformed_img_path=self.transformed_image_path,
            labels=y_train
        )

        val_ds = PTImageTextDataset(
            images=image_val,
            tokens=X_val,
            image_root_path=self.image_path,
            image_shape=self.image_shape,
            transformed_img_path=self.transformed_image_path,
            labels=y_val
        )

        test_ds = PTImageTextDataset(
            images=image_test,
            tokens=X_test,
            image_root_path=self.image_path,
            image_shape=self.image_shape,
            transformed_img_path=self.transformed_image_path,
            labels=y_test
        )

        self.train_dl = PTImageTextDataset.get_dataloader_from_dataset(train_ds, self.batch_size)
        self.val_dl = PTImageTextDataset.get_dataloader_from_dataset(val_ds, self.batch_size)
        self.test_dl = PTImageTextDataset.get_dataloader_from_dataset(test_ds, self.batch_size)

        return df_train, df_val, df_test

    def create_model(self):
        """
        It creates the model, compile and build the PyTorch model. This is just the combination of the trained image
        and text model. We connect these models with text and image processing layers, and a final prediction layer.

        The model is compiled with AdamW optimiser together with learning rate scheduler. It takes advantages of
        decreasing learning rate as well as the adaptive learning rate for each parameter in each optimisation steps.
        It uses categorical cross entropy as loss function and accuracy
        as the evaluation metric.

        A compiled model will be saved in the model attributes as a result.

        You may also find the model graph and summary in README of this project.

        """

        # we don't train the sequential layers here as it has been trained and fine-tuned in previous steps
        PTImageTextUtil.set_base_model_trainable(self.text_seq_layers, 0)
        PTImageTextUtil.set_base_model_trainable(self.image_seq_layers, 0)
        PTImageTextUtil.set_base_model_trainable(self.image_base_model, 0)
        PTImageTextUtil.set_base_model_trainable(self.text_embedding_layer, 0)

        self.model = PTImageTextModel(
            self.num_class,
            self.image_base_model,
            self.image_seq_layers,
            self.text_embedding_layer,
            self.text_seq_layers,
            self.is_transformer_based_text_model
        )

        self.model.to(self.device)
        print("Model created")

    def train_model(self) -> None:
        """
        Train the model with the training data. In each epoch, it will print out the loss and accuracy of the training
        and validation dataset in 'history' attribute. The records will be used for illustrating the performance of
        the model in later stage. There is a callback called tensorboard callback, which creates logs during the
        training process, and these logs can then be uploaded to TensorBoard.dev
        """

        optimizer, scheduler = prepare_optimizer_and_scheduler(
            self.model,
            len(self.train_dl.dataset),
            self.batch_size,
            self.learning_rate,
            self.epoch
        )

        self.writer = SummaryWriter(log_dir=self.log_path)

        self.history = train_and_validate_model(
            self.model,
            self.train_dl,
            self.val_dl,
            criterion=nn.CrossEntropyLoss(),
            epoch=self.epoch,
            optimizer=optimizer,
            scheduler=scheduler,
            summary_writer=self.writer
        )

    def evaluate_model(self) -> Tuple[float, float]:
        """
        Evaluate the model on the testing dataset. It returns only accuracy and loss to illustrate
        the overall performance of the model. If you need the prediction labels and the classification
        report returned, use predict_model instead.

        Returns:
            Tuple[float, float]: Loss and accuracy of the model.

        """

        return evaluate_model(
            self.model,
            self.test_dl,
            nn.CrossEntropyLoss(),
            self.classes
        )

    def predict_model(self, dataloader: Any) -> List[int]:
        """
        Predict with the model for records in the dataset.

        Args:
            dataloader (Optional, Any): DataLoader contains datasets with images, text and category.

        Returns:
            List[int]: List of labels

        """

        return predict_model(
            self.model, dataloader
        )
