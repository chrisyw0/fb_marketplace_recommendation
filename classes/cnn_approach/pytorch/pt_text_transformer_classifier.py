import pandas as pd
import torch
import torch.nn as nn

from typing import Tuple, List, Any
from dataclasses import field
from torch.utils.tensorboard import SummaryWriter

from classes.cnn_approach.pytorch.utils.pt_image_text_util import PTImageTextUtil
from classes.cnn_approach.pytorch.pt_base_classifier import (
    PTBaseClassifier,
    train_and_validate_model,
    evaluate_model,
    predict_model,
    prepare_optimizer_and_scheduler
)
from classes.cnn_approach.pytorch.utils.pt_dataset_generator import PTImageTextDataset
from classes.data_preparation.prepare_dataset import DatasetHelper


class PTTextTransformerClassifier(PTBaseClassifier):
    """
    A deep learning model predicting product category from its name and description.

    Args:
        df_image (pd.DataFrame): Image dataframe
        df_product (pd.DataFrame): Product dataframe

        embedding (str, Optional): The type of embedding model. Defaults to "BERT".

        embedding_dim (int, Optional): The vector size of embedding model. Defaults to 768.
        embedding_pretrain_model (str, Optional): Whether to use a pretrain model to encode the text. Please check
                                                  tf_text_processing_constant.py for available options.
                                                  Defaults to "bert_en_cased_L-12_H-768_A-12".
        batch_size (int, optional): Batch size of the model. Defaults to 16.
        dropout_pred (float, optional): Dropout rate of the layer before the prediction layer of the model.
                                              Defaults to 0.5.

        learning_rate (float, optional): Learning rate of the model. Defaults to 2e-5.
        epoch (float, optional): Epoch of the model. Defaults to 5

        metrics (List[str], optional):  list of metrics using for model evaluation. Defaults to ["accuracy"].

    """
    df_product: pd.DataFrame
    df_image: pd.DataFrame

    embedding: str = "BERT"

    embedding_dim: int = 768
    embedding_pretrain_model: str = "bert-base-cased"
    max_token_per_per_sentence: int = 512

    batch_size: int = 16
    dropout_pred: float = 0.5
    epoch: int = 5
    learning_rate: float = 2e-5

    metrics: List[str] = field(default_factory=lambda: ["accuracy"])

    def _get_model_name(self):
        return f"pt_text_model_{self.embedding}"

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

        print("Prepare training, validation and testing data")

        df_image_data['product_name_description'] = df_image_data["product_name_description"].apply(
            PTImageTextUtil.clean_text)

        # split dataset
        df_train, df_val, df_test = generator.split_dataset(df_image_data)

        X_train = df_train['product_name_description'].to_list()
        X_val = df_val['product_name_description'].to_list()
        X_test = df_test['product_name_description'].to_list()

        self.embedding_layer, tokenizer = PTImageTextUtil.prepare_embedding_model(
            embedding=self.embedding,
            embedding_dim=self.embedding_dim,
            pretrain_model=self.embedding_pretrain_model,
            trainable=False
        )

        X_train, X_val, X_test = PTImageTextUtil.batch_encode_text((X_train, X_val, X_test), tokenizer)

        # TODO: missing input shape and dtypes for model summary
        self.skip_summary = True
        # self.input_shape = {key: (self.max_token_per_per_sentence,) for key in encoded_text.keys()}
        # self.input_dtypes = [{key: torch.long for key in encoded_text.keys()}]

        y_train, y_val, y_test = generator.get_product_categories(df_train, df_val, df_test)

        train_ds = PTImageTextDataset(
            images=None,
            tokens=X_train,
            labels=y_train
        )

        val_ds = PTImageTextDataset(
            images=None,
            tokens=X_val,
            labels=y_val
        )

        test_ds = PTImageTextDataset(
            images=None,
            tokens=X_test,
            labels=y_test
        )

        self.train_dl = PTImageTextDataset.get_dataloader_from_dataset(train_ds, self.batch_size)
        self.val_dl = PTImageTextDataset.get_dataloader_from_dataset(val_ds, self.batch_size)
        self.test_dl = PTImageTextDataset.get_dataloader_from_dataset(test_ds, self.batch_size)

        return df_train, df_val, df_test

    def create_model(self) -> None:
        """
        Create the model with pretrained transformer based model for text classification. It includes following layers:
        - Preprocessing layer: This is the layer tokenising the text, returning the input ids and attention mask.
          The output is also padded with the same length to allow the tokens id inputting to the embedding layer
        - Embedding layer: This is the pre-trained embedding layer.
        - Dropout: Dropout a proportion of layers outputs from previous hidden layer
        - Dense: Linear layer with activation layer applied

        The input will be the pain text. To make the input shape the same for all products,
        padding is applied in previous function which simply append 0s to every product which has smaller tokens than
        the maximum, making the input shape to (batch_size, max number of tokens, 1).

        The output of the model will be the predicted probability of each class, which is equaled to
        (batch_size, num. of classes)

        The model is compiled with AdamW optimiser together with learning rate scheduler. It takes advantages of
        decreasing learning rate as well as the adaptive learning rate for each parameter in each optimisation steps.
        It uses categorical cross entropy as loss function and accuracy as the evaluation metric.

        This function will print out the summary of the model. You may also find the model graph and summary in README
        of this project.
        """

        PTImageTextUtil.set_base_model_trainable(
            self.embedding_layer, -1
        )

        class PTTextTransformerModel(nn.Module):
            def __init__(
                    self,
                    num_class,
                    embedding_dim,
                    embedding_layer,
                    dropout_pred
            ):
                super(PTTextTransformerModel, self).__init__()

                self.embedding_layer = embedding_layer

                self.sequential_layer = nn.Sequential(
                    nn.Linear(embedding_dim, 256),
                    nn.ReLU(),
                    nn.Dropout(dropout_pred)
                )

                self.prediction_layer = nn.Linear(256, num_class)

            def forward(self, text):
                x = self.embedding_layer(**text)["pooler_output"]
                x = self.sequential_layer(x)
                x = self.prediction_layer(x)
                return x

        self.model = PTTextTransformerModel(
            self.num_class,
            self.embedding_dim,
            self.embedding_layer,
            self.dropout_pred
        )

        self.model.to(self.device)
        print("Model created")

    def train_model(self) -> None:
        """
        Train the model with the training data. It applies early stop by monitoring loss of validation dataset.
        If the model fails to improve for 8 epochs, it will stop training to avoid overfitting.
        In each epoch, it will print out the loss and accuracy of the training and validation dataset
        in 'history' attribute. The records will be used for illustrating the performance of the model
        in later stage. There is a callback called tensorboard callback, which creates logs during the training process,
        and these logs can then be uploaded to TensorBoard.dev

        Since we don't have much other layers than the embedding layers, we can treat this step as the fine-tuning
        steps of the transformer based model.

        Returns:

        """

        print("=" * 80)
        print("Start training")
        print("=" * 80)

        optimizer, scheduler = prepare_optimizer_and_scheduler(
            self.model,
            len(self.train_dl.dataset.tokens["input_ids"]),
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

        print("Finish training")

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
            dataloader:

        Returns:
            List[int]: List of labels

        """

        return predict_model(
            self.model, dataloader
        )


    def save_model(self):
        """
        Save weight of the trained model.
        """

        super().save_model()
        torch.save(self.model.sequential_layer.state_dict(), f"{self.model_path}text_seq_layers.pt")
        torch.save(self.model.embedding_layer.state_dict(), f"{self.model_path}embedding_layer.pt")

    def load_model(self):
        """
        Load weight of the trained model.
        """
        super().load_model()
        self.model.sequential_layer.load_state_dict(torch.load(f"{self.model_path}text_seq_layers.pt"))
        self.model.embedding_layer.load_state_dict(torch.load(f"{self.model_path}image_base_model.pt"))
