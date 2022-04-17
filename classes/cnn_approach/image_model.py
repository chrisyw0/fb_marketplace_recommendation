from dataclasses import dataclass
import os
import tempfile
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from typing import Tuple, List
from sklearn.metrics import classification_report
from ..data_preparation.prepare_dataset import DatasetGenerator
from tensorboard.plugins.hparams import api as hp


@dataclass
class ImageModel:
    """
    A deep learning model predicting product category of an image. The correct flow of training and
    testing the model is as follows:

    1. prepare_data - Input product and image data and get the training, validation and testing dataset.
    2. create_model - Create a deep learning model according to the input data shape and other configuration
    3. train_model - Train the model with training dataset, and validate it with validation dataset.
    4. evaluate_model/predict_model - Test the model with the testing dataset. evaluate_model will only get the
                                      overall accuracy and loss while predict_model will return a classification
                                      report and predicted labels for the testing dataset.
    5. visualise_performance - Plot the accuracy and loss in each epoch for training and validation dataset
    6. save_model - Save the weight for the model for later use.
    7. clean_up - remove the folders storing the images.

    TODO: reuse the model

    Args:
        df_image (pd.DataFrame): Image dataframe
        df_product (pd.DataFrame): Product dataframe
        image_path (str, optional): Path to cache the image dataframe. Defaults to "./data/images/".
        log_path (str, optional): Path to cache the training logs. Defaults to "./data/logs/".
        batch_size (int, optional): Batch size of the model Defaults to 32.
        input_shape (Tuple[int, int, int], Optional): Size of the image inputting to the model.
                                                      If image channel = 'RGB', the value will be
                                                      (width, height, 3) i.e. 3 channels
                                                      Defaults to (256, 256, 3)
        dropout (float, optional): Dropout rate of the model Defaults to 0.2.
        learning_rate (float, optional): Learning rate of the model Defaults to 0.01.
        epoch (float, optional): Epoch of the model Defaults to 10.

    """
    df_product: pd.DataFrame
    df_image: pd.DataFrame

    image_path: str = "./data/images/"
    log_path: str = "./data/logs/"

    batch_size = 32
    input_shape: Tuple[int, int, int] = (256, 256, 3)
    dropout: float = 0.6
    learning_rate: float = 0.01

    epoch: int = 15

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
        # make comparsion of the metrics

        generator = DatasetGenerator(self.df_product, self.df_image)
        df_image_data = generator.generate_image_product_dataset()

        self.num_class = len(pd.unique(df_image_data["root_category"]))

        # create temp folders
        self.train_tmp_folder = tempfile.mkdtemp()
        self.val_tmp_folder = tempfile.mkdtemp()
        self.test_tmp_folder = tempfile.mkdtemp()

        print(f"Training data path = {self.train_tmp_folder}")
        print(f"Validation data path = {self.val_tmp_folder}")
        print(f"Testing data path = {self.test_tmp_folder}")

        df_train, df_val, df_test = generator.split_dataset(df_image_data)

        # assign the images for training, validation and testing dataset
        idx = 0

        train_idx = df_train.index
        val_idx = df_val.index
        test_idx = df_test.index

        # walk through the images in the image path, assign each image into
        # training, validation and testing folders, and save a list of its category
        for root, dirs, files in os.walk(self.image_path):
            for file in files:
                product_id = file.split(".")[0]
                category = df_image_data[df_image_data["id_x"] == product_id]["root_category"]

                if category.size > 0:
                    if idx in train_idx:
                        os.makedirs(f"{self.train_tmp_folder}/{category.iloc[0]}", exist_ok=True)
                        shutil.copy2(f"{self.image_path}{file}", f"{self.train_tmp_folder}/{category.iloc[0]}/{file}")
                    elif idx in val_idx:
                        os.makedirs(f"{self.val_tmp_folder}/{category.iloc[0]}", exist_ok=True)
                        shutil.copy2(f"{self.image_path}{file}", f"{self.val_tmp_folder}/{category.iloc[0]}/{file}")
                    else:
                        os.makedirs(f"{self.test_tmp_folder}/{category.iloc[0]}", exist_ok=True)
                        shutil.copy2(f"{self.image_path}{file}", f"{self.test_tmp_folder}/{category.iloc[0]}/{file}")

                    idx += 1

        # create a tensorflow dataset
        self.ds_train = tf.keras.utils.image_dataset_from_directory(
            self.train_tmp_folder,
            label_mode="categorical",
            color_mode="rgb",
            batch_size=self.batch_size,
            follow_links=True
        )

        self.ds_val = tf.keras.utils.image_dataset_from_directory(
            self.val_tmp_folder,
            label_mode="categorical",
            color_mode="rgb",
            batch_size=self.batch_size,
            follow_links=True
        )

        self.ds_test = tf.keras.utils.image_dataset_from_directory(
            self.test_tmp_folder,
            label_mode="categorical",
            color_mode="rgb",
            batch_size=self.batch_size,
            follow_links=True,
            shuffle=False
        )

        return df_image_data.iloc[train_idx], df_image_data.iloc[val_idx], df_image_data.iloc[test_idx]

    def create_model(self) -> None:
        """
        Create an CNN model based on RestNet50V2. The model consists of preprocessing layer,
        data augmentation layer, global averaging layer, RestNet50V2 model, dropout layer and
        finally the prediction layer. The input shape can be configured in the class attributes
        input_shape and the output size will be equal to number of classes (determined in the
        prepare_data function).

        The model is compiled with RMSprop optimiser which reduces learning rate based on the loss
        history in the training stage. It uses categorical cross entropy as loss function and accuracy
        as the evaluation metric.

        A compiled model will be saved in the model attributes as a result.

        """

        # pre-process the image to make the each value [-1, 1]
        preprocess_input = tf.keras.applications.resnet_v2.preprocess_input

        # use the based model RestNet50 V2
        base_model = tf.keras.applications.ResNet50V2(include_top=False, input_shape=self.input_shape)
        base_model.trainable = False

        print(base_model.summary())

        # add layers for the model:
        # data augmentation, pooling layer, dropout,
        # dense layer and prediction layer

        image_batch, label_batch = next(iter(self.ds_train))
        feature_batch = base_model(image_batch)

        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        feature_batch_average = global_average_layer(feature_batch)
        
        dense_layer = tf.keras.layers.Dense(128, activation="relu")
        dense_patch = dense_layer(feature_batch_average)

        prediction_layer = tf.keras.layers.Dense(self.num_class, activation="softmax")
        _ = prediction_layer(dense_patch)

        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip('horizontal'),
            tf.keras.layers.RandomRotation(0.2),
        ])

        inputs = tf.keras.Input(shape=self.input_shape)
        x = data_augmentation(inputs)
        x = preprocess_input(x)
        x = base_model(x, training=False)
        x = global_average_layer(x)
        x = tf.keras.layers.Dropout(self.dropout)(x)
        x = dense_layer(x)
        outputs = prediction_layer(x)

        # combine the model and print model summary
        self.model = tf.keras.Model(inputs, outputs)
        self.model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate / 10),
                           loss=tf.keras.losses.CategoricalCrossentropy(),
                           metrics=['accuracy'])

        print(self.model.summary())

    def train_model(self) -> None:
        """
        Train a model with the training data. In each epoch, it will print out the loss and accuracy
        of the training and validation dataset in 'history' attribute. The records will be used for
        illustrating the performance of the model in later stage. There are two callbacks called tensorboard callback
        and hyperparameter call back, it will create logs during the training process, and these logs can then be
        uploaded to TensorBoard.dev

        """

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=self.log_path, histogram_freq=1)

        hparams_callback = hp.KerasCallback(self.log_path, {
            'dropout': self.dropout
        })

        self.history = self.model.fit(self.ds_train,
                                      epochs=self.epoch,
                                      validation_data=self.ds_val,
                                      callbacks=[tensorboard_callback, hparams_callback])

    def evaluate_model(self, dataset: tf.data.Dataset = None) -> Tuple[float, float]:
        """
        Evaluate the model on a dataset. It returns only accuracy and loss to illustrate the overall performance
        of the model. If you need the prediction labels and the classification report returned, use predict_model
        instead.

        Args:
            dataset (Optional, tf.data.Dataset): Dataset to be used to evaluate the model. If None is input,
                                                 the testing dataset set in prepare_data will be used. Defaults to
                                                 None.

        Returns:
            Tuple[float, float]: Loss and accuracy of the model.

        """
        if dataset == None:
            dataset = self.ds_test

        loss, accuracy = self.model.evaluate(dataset)
        return loss, accuracy

    def predict_model(self, dataset: tf.data.Dataset = None) -> Tuple[List[int], classification_report]:
        """
        Predict with the model for records in a dataset. It returns a classification report and prediction result
        of a given dataset. If you want to have accuracy and loss to evaluate the overall performance of the model,
        use evaluate_model instead.

        Args:
            dataset (Optional, tf.data.Dataset): Dataset to be used to evaluate the model. If None is input,
                                                 the testing dataset set in prepare_data will be used. Defaults to
                                                 None.

        Returns:
            Tuple[List[int], classification_report]: List of labels and classification report

        """
        if dataset == None:
            dataset = self.ds_test

        prediction = self.model.predict(dataset)

        y_true = [np.argmax(z) for z in np.concatenate([y for x, y in dataset])]
        y_pred = [np.argmax(x) for x in prediction]

        report = classification_report(y_true, y_pred)
        print(report)

        return y_pred, report

    def load_model(self, path: str = "./data/model/"):
        """
        Create a model with saved weight
        Args:
            path (str, Optional): Path for the weights, Defaults to ./data/model/

        """

        self.create_model()
        self.model.load_weights(f"{path}model.ckpt")

    def save_model(self, path: str = "./data/model/"):
        """
        Save weight of the trained model.
        Args:
            path (str, Optional): Path for the weights, Defaults to ./data/model/

        """

        self.model.save_weights(f"{path}model.ckpt")

    def visualise_performance(self) -> None:
        """
        Visual the performance of the model. It will plot loss and accuracy for training and validation dataset
        in each epoch.

        """
        # plot the loss
        plt.plot(self.history.history['loss'], label='train loss')
        plt.plot(self.history.history['val_loss'], label='val loss')
        plt.legend()
        plt.show()

        # plot the accuracy
        plt.plot(self.history.history['accuracy'], label='train acc')
        plt.plot(self.history.history['val_accuracy'], label='val acc')
        plt.legend()
        plt.show()

    def clean_up(self) -> None:
        """
        Remove the folders storing the images.
        """

        shutil.rmtree(self.train_tmp_folder)
        shutil.rmtree(self.val_tmp_folder)
        shutil.rmtree(self.test_tmp_folder)
