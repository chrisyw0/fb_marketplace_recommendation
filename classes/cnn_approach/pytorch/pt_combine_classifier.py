import pandas as pd
import torch
import torch.nn as nn
import copy

from typing import Tuple, List, Any, Optional
from dataclasses import field
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from classes.cnn_approach.pytorch.utils.pt_image_text_util import PTImageTextUtil
from .pt_base_classifier import (
    PTBaseClassifier,
    train_and_validate_model,
    evaluate_model,
    predict_model,
    prepare_optimizer_and_scheduler
)

from classes.cnn_approach.pytorch.utils.pt_dataset_generator import PTImageTextDataset
from classes.data_preparation.prepare_dataset import DatasetHelper


class PTImageTextClassifier(PTBaseClassifier):
    """
    A deep learning model predicting product category from its image, name and description.

    Args:
        df_image (pd.DataFrame): Image dataframe
        df_product (pd.DataFrame): Product dataframe

        image_seq_layers (tf.keras.Model): The trained image model (without preprocessing and prediction layer).
        text_seq_layers (tf.keras.Model): The trained text model (without preprocessing and prediction layer).

        image_base_model_name (str, optional): Name of the image pre-trained model. Defaults to "EfficientNetB3"
        embedding (str, Optional): The type of embedding model. Defaults to "Word2Vec".

        embedding_pretrain_model (str, Optional): Whether to use a pretrain model to encode the text.
                                                  Defaults to None, which means no pretrained model is used.

        transformed_image_path (str, optional): Path to cache the transformed image. This is improving when training,
                                                validating and testing the model as we don't need to transform and resize
                                                images when they are loaded into memory. Defaults to "./data/adjusted_img/"

        input_shape (Tuple[int, int, int], Optional): Size of the image inputting to the model.
                                                      If image channel = 'RGB', the value will be
                                                      (width, height, 3) i.e. 3 channels
                                                      Defaults to (300, 300, 3)

        epoch (float, optional): Epoch of the model. Defaults to 3.
        learning_rate (float, optional): Learning rate of the model. Defaults to 1e-6.
        batch_size (int, optional): Batch size of the model. Defaults to 16.

        metrics (List[str], optional):  list of metrics using for model evaluation. Defaults to ["accuracy"].


    """
    df_product: pd.DataFrame
    df_image: pd.DataFrame

    image_base_model_name: str = "EfficientNetB3"
    embedding: str = "BERT"
    embedding_dim: int = 768
    max_token_per_per_sentence: int = 512

    image_path: str = "./data/images/"
    transformed_image_path: str = "./data/adjusted_img/"

    embedding_pretrain_model: str = "bert-base-cased"

    image_shape: Tuple[int, int, int] = (300, 300, 3)
    transformed_image_path += str(image_shape[0]) + "/"

    epoch: int = 3
    learning_rate: float = 1e-3
    batch_size: int = 16

    metrics: List[str] = field(default_factory=lambda: ["accuracy"])

    def __init__(self,
                 df_image: pd.DataFrame,
                 df_product: pd.DataFrame,
                 image_base_model: Any,
                 image_seq_layers: Any,
                 text_seq_layers: Any,
                 is_transformer_based_text_model,
                 embedding_model: Optional[Any]
                 ):

        super().__init__(df_image, df_product)

        self.image_seq_layers = copy.deepcopy(image_seq_layers)
        self.text_seq_layers = copy.deepcopy(text_seq_layers)
        self.image_base_model = copy.deepcopy(image_base_model)
        self.is_transformer_based_text_model = is_transformer_based_text_model

        self.embedding_model = embedding_model

    def _get_model_name(self):
        return f"pt_image_text_model_{self.image_base_model_name}_{self.embedding}"

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare training, validation and testing data for the model. It includes building the word embedding model,
        splitting the dataset and getting essential elements for later stages.

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

            print(f"Creating a {self.embedding} model, " \
                  f"dimension {self.embedding_dim}, " \
                  f"pre-train model {self.embedding_pretrain_model}")

            # create and train the embedding model
            self.embedding_model = PTImageTextUtil.prepare_embedding_model(
                embedding=self.embedding,
                embedding_dim=self.embedding_dim,
                training_data=training_data,
                pretrain_model=self.embedding_pretrain_model
            )

            self.text_embedding_layer = None

            print("Getting index from the embedding model")

            # convert token into token index in embedding model weight matrix
            df_image_data['tokens_index'] = PTImageTextUtil.get_token_index(
                product_tokens,
                self.embedding,
                self.embedding_model
            )

            self.input_shape = {
                "text": (self.num_max_tokens,),
                "image": (self.image_shape[2], self.image_shape[1], self.image_shape[0])
            }
            self.input_dtypes = [{"text": torch.int, "image": torch.float}]

            print("Prepare training, validation and testing data")

            # split dataset
            df_train, df_val, df_test = generator.split_dataset(df_image_data)

            X_train = df_train['tokens_index'].to_list()
            X_val = df_val['tokens_index'].to_list()
            X_test = df_test['tokens_index'].to_list()

            X_train, X_val, X_test = PTImageTextUtil.prepare_token_with_padding((X_train, X_val, X_test))

        else:
            df_image_data['product_name_description'] = df_image_data["product_name_description"].apply(
                PTImageTextUtil.clean_text)

            self.text_embedding_layer, tokenizer = PTImageTextUtil.prepare_embedding_model(
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

            # TODO: missing input shape and dtypes for model summary
            self.skip_summary = True
            self.input_shape = {key: (self.max_token_per_per_sentence,) for key in X_train.keys()}
            self.input_dtypes = [{key: torch.long for key in X_train.keys()}]


        print("Prepare training, validation and testing data")

        image_train, image_val, image_test = generator.get_image_ids(df_train, df_val, df_test)
        y_train, y_val, y_test = generator.get_product_categories(df_train, df_val, df_test)

        train_ds = PTImageTextDataset(
            images=image_train,
            tokens=X_train,
            image_root_path=self.image_path,
            image_shape=self.image_shape,
            temp_img_path=self.transformed_image_path,
            labels=y_train
        )

        val_ds = PTImageTextDataset(
            images=image_val,
            tokens=X_val,
            image_root_path=self.image_path,
            image_shape=self.image_shape,
            temp_img_path=self.transformed_image_path,
            labels=y_val
        )

        test_ds = PTImageTextDataset(
            images=image_test,
            tokens=X_test,
            image_root_path=self.image_path,
            image_shape=self.image_shape,
            temp_img_path=self.transformed_image_path,
            labels=y_test
        )

        self.train_dl = PTImageTextDataset.get_dataloader_from_dataset(train_ds, self.batch_size)
        self.val_dl = PTImageTextDataset.get_dataloader_from_dataset(val_ds, self.batch_size)
        self.test_dl = PTImageTextDataset.get_dataloader_from_dataset(test_ds, self.batch_size)

        return df_train, df_val, df_test

    def create_model(self) -> None:
        """
        It creates the model, compile and build the tensorflow model. This is just the combination of image and
        text model. We connect these models with text and image processing layers, and a final prediction layer.

        The model is compiled with AdamW optimiser together with learning rate scheduler. It takes advantages of
        decreasing learning rate as well as the adaptive learning rate for each parameter in each optimisation steps.
        It uses categorical cross entropy as loss function and accuracy
        as the evaluation metric.

        A compiled model will be saved in the model attributes as a result.

        This function will print out the summary of the model. You may also find the model graph and summary in README
        of this project.

        """

        # we don't train the sequential layers here as it has been trained and fine-tuned in previous steps
        PTImageTextUtil.set_base_model_trainable(self.text_seq_layers, -1)
        PTImageTextUtil.set_base_model_trainable(self.image_seq_layers, -1)

        class PTImageTextModel(nn.Module):
            def __init__(
                    self,
                    num_class,
                    image_base_model,
                    image_seq_layers,
                    text_embedding_layer,
                    text_seq_layers
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

                self.prediction_layer = nn.Linear(512, num_class)

            def forward(self, image, text):
                x_img = self.transforms(image)
                x_img = self.image_base_model(x_img)
                x_img = x_img.squeeze()
                if x_img.dim() == 1:
                    # if only one data in a batch, it add back the dimension
                    x_img = x_img.unsqueeze(0)
                x_img = self.image_seq_layers(x_img)

                if self.text_embedding_layer:
                    x_text = self.text_embedding_layer(**text)["pooler_output"]
                else:
                    x_text = text

                x_text = self.text_seq_layers(x_text)

                x = torch.cat([x_img, x_text], dim=1)
                x = self.prediction_layer(x)

                return x

        self.model = PTImageTextModel(
            self.num_class,
            self.image_base_model,
            self.image_seq_layers,
            self.text_embedding_layer,
            self.text_seq_layers
        )

        self.model.to(self.device)
        print("Model created")

    def train_model(self) -> None:
        """
        Train the model with the training data. It applies early stop by monitoring loss of validation dataset.
        In each epoch, it will print out the loss and accuracy of the training and validation dataset
        in 'history' attribute. The records will be used for illustrating the performance of the model
        in later stage. There is a callback called tensorboard callback, which creates logs during the training process,
        and these logs can then be uploaded to TensorBoard.dev
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
