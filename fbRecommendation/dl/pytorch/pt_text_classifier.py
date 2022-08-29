import pandas as pd
import torch
import torch.nn as nn

from typing import Tuple, List, Any
from dataclasses import field

from torch.utils.tensorboard import SummaryWriter

from fbRecommendation.dl.pytorch.utils.pt_image_text_util import PTImageTextUtil
from fbRecommendation.dl.pytorch.pt_base_classifier import (
    PTBaseClassifier,
    train_and_validate_model,
    evaluate_model,
    predict_model,
    prepare_optimizer_and_scheduler
)
from fbRecommendation.dl.pytorch.utils.pt_dataset_generator import PTImageTextDataset
from fbRecommendation.dataset.prepare_dataset import DatasetHelper


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


class PTTextClassifier(PTBaseClassifier):
    """
    A deep learning model predicting product category from its name and description.

    Args:
        df_image (pd.DataFrame): Image dataframe
        df_product (pd.DataFrame): Product dataframe

        embedding (str, Optional): The type of embedding model. Defaults to "Word2Vec".

        embedding_dim (int, Optional): The vector size of embedding model. Defaults to 300.
        embedding_pretrain_model (str, Optional): Whether to use a pretrain model to encode the text.
                                                  Defaults to None, which means no pretrained model is used.

        batch_size (int, optional): Batch size of the model. Defaults to 32.

        dropout_conv (float, optional): Dropout rate of the convolution layer of the model. Defaults to 0.5.
        dropout_prediction (float, optional): Dropout rate of the layer before the prediction layer of the model.
                                              Defaults to 0.3.

        learning_rate (float, optional): Learning rate of the model. Defaults to 0.001.
        epoch (float, optional): Epoch of the model. Defaults to 15.

        fine_tune_base_model (bool, optional): Whether fine-tuning model is required. Default to True.
        fine_tune_learning_rate (float, optional): Learning rate of the model in the fine-tuning stage.
                                                   Defaults to 0.001.
        fine_tune_epoch: (int, optional): Number of epochs in fine-tuning stage. Defaults to 15.

        metrics (List[str], optional):  list of metrics using for model evaluation. Defaults to ["accuracy"].

    """
    df_product: pd.DataFrame
    df_image: pd.DataFrame

    embedding: str = "Word2Vec"

    embedding_dim: int = 300
    embedding_pretrain_model: str = None

    batch_size: int = 16
    dropout_conv: float = 0.5
    dropout_pred: float = 0.3
    learning_rate: float = 0.001
    epoch: int = 15

    fine_tune_base_model: bool = True
    fine_tune_learning_rate: float = 0.001
    fine_tune_epoch: int = 15

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

        print("Getting index from the embedding model")

        # convert token into token index in embedding model weight matrix
        df_image_data['tokens_index'] = PTImageTextUtil.get_token_index(
            product_tokens,
            self.embedding,
            self.embedding_model
        )

        print("Prepare training, validation and testing data")

        self.input_shape = (self.num_max_tokens,)
        self.input_dtypes = [torch.int]

        # split dataset
        df_train, df_val, df_test = generator.split_dataset(df_image_data)

        X_train = df_train['tokens_index'].to_list()
        X_val = df_val['tokens_index'].to_list()
        X_test = df_test['tokens_index'].to_list()

        X_train, X_val, X_test = PTImageTextUtil.prepare_token_with_padding(
            (X_train, X_val, X_test),
            self.embedding
        )
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
        Create the CNN model for text classification. It includes following layers:
        - Embedding layer: This layer is a layer with pre-built weights copied from embedding layer.
        - Conv1D: Convolution layer with activation layer applied
        - AveragePooling1D: Averaging pooling layer
        - Flatten: Flatten the 2D array input to 1D array
        - Dropout: Dropout a proportion of layers outputs from previous hidden layer
        - Linear: Linear layer with activation layer applied

        The input will be the token index for each product's tokens. To make the input shape the same for all products,
        padding is applied in previous function which simply append 0s to every product which has smaller tokens than
        the maximum, making the input shape to (batch_size, max number of tokens, 1).

        The output of the model will be the predicted probability of each class, which is equaled to
        (batch_size, num. of fbRecommendation)

        The model is compiled with AdamW optimiser together with learning rate scheduler. It takes advantages of
        decreasing learning rate as well as the adaptive learning rate for each parameter in each optimisation steps.
        It uses categorical cross entropy as loss function and accuracy as the evaluation metric.

        You may also find the model graph and summary in README of this project.

        """

        self.embedding_layer = PTImageTextUtil.gensim_to_pytorch_embedding(
            self.embedding_model)

        self.model = PTTextModel(
            self.num_class,
            self.embedding_dim,
            self.embedding_layer,
            self.dropout_conv,
            self.dropout_pred
        )

        self.model.to(self.device)
        print("Model created")

    def train_model(self) -> None:
        """
        Train the model with the training data. It applies early stop by monitoring loss of validation dataset.
        If the model fails to improve for 5 epochs, it will stop training to avoid overfitting.
        In each epoch, it will print out the loss and accuracy of the training and validation dataset
        in 'history' attribute. The records will be used for illustrating the performance of the model
        in later stage. There is a callback called tensorboard callback which create logs during the training process,
        and these logs can then be uploaded to TensorBoard.dev

        Returns:

        """
        print("=" * 80)
        print("Start training")
        print("=" * 80)

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
            summary_writer=self.writer,
            early_stop_count=5
        )

        print("Finish training")

    def fine_tune_model(self) -> None:
        """
        Fine-tuning the model by unfreeze the embedding. Our Word2Vec model is freeze in the training stage and
        not optimised for our final prediction. This fune-tuning stage will fine-tune the weights of all layers with
        lower learning rate to improve the performance of this model. It again applies early stop by monitoring loss of
        validation dataset. If the model fails to improve for 5 epochs, it will stop training to avoid overfitting.
        It uses the same components for training and validation. The result will be appended in to the
        history attribute.
        """

        if self.fine_tune_base_model:
            print("=" * 80)
            print("Start fine-tuning")
            print("=" * 80)

            PTImageTextUtil.set_base_model_trainable(
                self.model.embedding_layer, -1
            )

            optimizer, scheduler = prepare_optimizer_and_scheduler(
                self.model,
                len(self.train_dl.dataset),
                self.batch_size,
                self.fine_tune_learning_rate,
                self.fine_tune_epoch
            )

            # For some reason, the val accuracy keeps increasing even the val_loss increasing in last few epoch. This
            # may happen there are some outliner records that the model can't recognise, but the model actually improve
            # on other normal records.
            fine_tune_history = train_and_validate_model(
                self.model,
                self.train_dl,
                self.val_dl,
                criterion=nn.CrossEntropyLoss(),
                epoch=self.fine_tune_epoch,
                optimizer=optimizer,
                scheduler=scheduler,
                summary_writer=self.writer,
                init_epoch=self.epoch + 1,
                early_stop_count=5,
                early_stop_metric="accuracy",
                restore_weight=False
            )

            self.history['loss'].extend(fine_tune_history['loss'])
            self.history['val_loss'].extend(fine_tune_history['val_loss'])
            self.history['accuracy'].extend(fine_tune_history['accuracy'])
            self.history['val_accuracy'].extend(fine_tune_history['val_accuracy'])

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
        self.model.embedding_layer.load_state_dict(torch.load(f"{self.model_path}embedding_layer.pt"))
