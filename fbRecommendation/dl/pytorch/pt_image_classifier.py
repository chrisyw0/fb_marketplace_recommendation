import pandas as pd
import torch
import torch.nn as nn

from typing import Tuple, List, Any
from dataclasses import field
from torch.utils.tensorboard import SummaryWriter

from fbRecommendation.dataset.prepare_dataset import DatasetHelper
from .pt_base_classifier import (
    PTBaseClassifier,
    train_and_validate_model,
    evaluate_model,
    predict_model,
    prepare_optimizer_and_scheduler
)
from fbRecommendation.dl.pytorch.utils.pt_dataset_generator import PTImageTextDataset
from fbRecommendation.dl.pytorch.model.pt_model_util import PTModelUtil
from fbRecommendation.dl.pytorch.model.pt_model import PTImageModel


class PTImageClassifier(PTBaseClassifier):
    """
    A deep learning model predicting product category of an image.

    Args:
        df_image (pd.DataFrame): Image dataframe
        df_product (pd.DataFrame): Product dataframe

        image_base_model (str, optional): Name of the image pre-trained model

        image_path (str, optional): Path to cache the image dataframe. Defaults to "./data/images/".
        transformed_image_path (str, optional): Path to save the preprocessed images.
                                                Defaults to "./data/adjusted_img/" + image_shape[0].

        batch_size (int, optional): Batch size of the model Defaults to 12.
        image_shape (Tuple[int, int, int], Optional): Size of the image inputting to the model.
                                                      If image channel = 'RGB', the value will be
                                                      (width, height, 3) i.e. 3 channels
                                                      Defaults to (300, 300, 3)
        dropout_conv (float, optional): Dropout rate of the convolution layer of the model. Defaults to 0.6.
        dropout_pred (float, optional): Dropout rate of the layer before the prediction layer of the model.
                                        Defaults to 0.3.

        learning_rate (float, optional): Learning rate of the model in the training stage. Defaults to 1e-3.
        epoch (float, optional): Epoch of the model Defaults to 12.

        fine_tune_base_model (bool, optional): Whether fine tuning model is required. Defaults to True.
        fine_tune_base_model_layers (int, optional): Number of layers to be fine-tuned in the fune tuning stage.
                                                     -1 means all layers will be unfreezed. Defaults to -1
        fine_tune_learning_rate (float, optional): Learning rate of the model in the fine-tuning stage.
                                                   Defaults to 1e-5.
        fine_tune_epoch: (int, optional): Number of epochs in fine-tuning stage. Defaults to 8.

        metrics (List[str], optional):  list of metrics using for model evaluation. Defaults to ["accuracy"].

    """
    df_product: pd.DataFrame
    df_image: pd.DataFrame

    image_base_model: str = "RestNet50"

    image_path: str = "./data/images/"
    transformed_image_path: str = "./data/adjusted_img/"

    batch_size = 12
    image_shape: Tuple[int, int, int] = (300, 300, 3)
    dropout_conv: float = 0.6
    dropout_pred: float = 0.3

    learning_rate: float = 1e-3
    epoch: int = 12

    fine_tune_base_model: bool = True
    fine_tune_base_model_layers: int = -1
    fine_tune_learning_rate: float = 1e-5
    fine_tune_epoch: int = 8

    transformed_image_path += str(image_shape[0]) + "/"

    metrics: List[str] = field(default_factory=lambda: ["accuracy"])

    def _get_model_name(self):
        return f"pt_image_model_{self.image_base_model}"

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for the model, by merging the image and product dataframe,
        assigning images into folder for tensorflow dataset and
        gathering labels (category) of from the dataframe

        Returns: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            - dataframe of training, validation and testing dataset

        """

        # split the dataset into training, validation and testing dataset
        # it uses the same function as the machine learning model so it could
        # make comparison of the metrics
        generator = DatasetHelper(self.df_product, self.df_image)
        df_image_data, self.classes = generator.generate_image_product_dataset()

        self.num_class = len(self.classes)

        self.input_shape = (self.image_shape[2], self.image_shape[1], self.image_shape[0])
        self.input_dtypes = [torch.float]

        # split dataset
        df_train, df_val, df_test = generator.split_dataset(df_image_data)
        y_train, y_val, y_test = generator.get_product_categories(df_train, df_val, df_test)
        image_train, image_val, image_test = generator.get_image_ids(df_train, df_val, df_test)

        train_ds = PTImageTextDataset(
            images=image_train,
            tokens=None,
            image_root_path=self.image_path,
            image_shape=self.image_shape,
            transformed_img_path=self.transformed_image_path,
            labels=y_train
        )

        val_ds = PTImageTextDataset(
            images=image_val,
            tokens=None,
            image_root_path=self.image_path,
            image_shape=self.image_shape,
            transformed_img_path=self.transformed_image_path,
            labels=y_val
        )

        test_ds = PTImageTextDataset(
            images=image_test,
            tokens=None,
            image_root_path=self.image_path,
            image_shape=self.image_shape,
            transformed_img_path=self.transformed_image_path,
            labels=y_test
        )

        self.train_dl = PTImageTextDataset.get_dataloader_from_dataset(train_ds, self.batch_size)
        self.val_dl = PTImageTextDataset.get_dataloader_from_dataset(val_ds, self.batch_size)
        self.test_dl = PTImageTextDataset.get_dataloader_from_dataset(test_ds, self.batch_size)

        return df_train, df_val, df_test

    def create_model(self) -> None:
        """
        Create an CNN model based on RestNet50V2 or EfficientNetV2. The model consists of preprocessing layer,
        data augmentation layer, global averaging layer, RestNet50V2 model, dropout layer and
        finally the prediction layer. The input shape can be configured in the class attributes
        input_shape and the output size will be equal to number of fbRecommendation (determined in the
        prepare_data function).

        The model is compiled with AdamW optimiser together with learning rate scheduler. It takes advantages of
        decreasing learning rate as well as the adaptive learning rate for each parameter in each optimisation steps.
        It uses categorical cross entropy as loss function and accuracy
        as the evaluation metric.

        A compiled model will be saved in the model attributes as a result.

        You may also find the model graph and summary in README of this project.
        """

        base_model_dim = {
            "EfficientNetB0": 1280,
            "EfficientNetB3": 1280,
            "RestNet50": 2048
        }

        self.model = PTImageModel(
            self.num_class,
            self.image_base_model,
            self.image_shape,
            self.dropout_conv,
            self.dropout_pred,
            base_model_dim.get(self.image_base_model, 2048)
        )

        self.model.to(self.device)

    def train_model(self) -> None:
        """
        Train a model with the training data. It applies early stop by monitoring loss of validation dataset.
        If the model fails to improve for 5 epochs, it will stop training to avoid overfitting.
        In each epoch, it will print out the loss and accuracy of the training and validation dataset in 'history'
        attribute. The records will be used for illustrating the performance of the model in later stage.
        There are 2 callbacks called early_stop_callback, tensorboard callback: early_stop_callback will detect whether
        the model is overfitted and stop training while tensorboard callback will create logs during the training
        process, and these logs can then be uploaded to TensorBoard.dev.

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

    def fine_tune_model(self) -> None:
        """
        Fine-tuning the model by unfreeze the image based model. Since the based model is usually pre-trained with a
        larger dataset, it will be better for us not to change the weights significantly, or we will lose the power of
        transfer learning. It again applies early stop by monitoring loss of validation dataset.
        If the model fails to improve for 5 epochs, it will stop training to avoid overfitting.
        It uses the same components for training and validation. The result will be appended in to
        the history attribute.

        """
        if self.fine_tune_base_model:
            print("=" * 80)
            print("Start fine-tuning")
            print("=" * 80)

            PTModelUtil.set_base_model_trainable(
                self.model.image_base_model,
                self.fine_tune_base_model_layers
            )

            optimizer, scheduler = prepare_optimizer_and_scheduler(
                self.model,
                len(self.train_dl.dataset),
                self.batch_size,
                self.fine_tune_learning_rate,
                self.fine_tune_epoch
            )

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
                early_stop_count=5
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
        Predict with the model for records in a dataset.

        Args:
            dataloader (Optional, Any): DataLoader contains datasets with images and category.

        Returns:
            List[int]: List of labels.

        """

        return predict_model(
            self.model, dataloader
        )

    def save_model(self):
        """
        Save weight of the trained model.
        """

        super().save_model()
        torch.save(self.model.sequential_layer.state_dict(), f"{self.model_path}image_seq_layers.pt")
        torch.save(self.model.image_base_model.state_dict(), f"{self.model_path}image_base_model.pt")

    def load_model(self):
        """
        Load weight of the trained model.
        """
        super().load_model()
        self.model.sequential_layer.load_state_dict(torch.load(f"{self.model_path}image_seq_layers.pt"))
        self.model.image_base_model.load_state_dict(torch.load(f"{self.model_path}image_base_model.pt"))
