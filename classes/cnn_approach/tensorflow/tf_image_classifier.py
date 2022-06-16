import pandas as pd
import tensorflow as tf
import numpy as np
import math

from typing import Tuple, List, Any
from sklearn.metrics import classification_report
from sklearn import preprocessing
from tensorboard.plugins.hparams import api as hp
from dataclasses import field
from official.nlp import optimization

from classes.data_preparation.prepare_dataset import DatasetHelper
from .tf_base_classifier import TFBaseClassifier
from classes.cnn_approach.tensorflow.utils.tf_image_text_util import TFImageTextUtil
from classes.cnn_approach.tensorflow.utils.tf_dataset_generator import TFDatasetGenerator


class TFImageClassifier(TFBaseClassifier):
    """
    A deep learning model predicting product category of an image.

    Args:
        df_image (pd.DataFrame): Image dataframe
        df_product (pd.DataFrame): Product dataframe

        model_name (str, optional): Name of the model.
        image_base_model (str, optional): Name of the image pre-trained model

        image_path (str, optional): Path to cache the image dataframe. Defaults to "./data/images/".
        log_path (str, optional): Path to cache the training logs. Defaults to "./logs/image_model/".
        model_path (str, optional): Path to cache the weight of the image model.
                                    Defaults to "./model/image_model/weights/".
        transformed_image_path (str, optional): Path to save the preprocessed images.
                                                Defaults to "./data/adjusted_img/" + image_shape[0].

        batch_size (int, optional): Batch size of the model Defaults to 32.
        image_shape (Tuple[int, int, int], Optional): Size of the image inputting to the model.
                                                      If image channel = 'RGB', the value will be
                                                      (width, height, 3) i.e. 3 channels
                                                      Defaults to (300, 300, 3)
        dropout_conv (float, optional): Dropout rate of the convolution layer of the model. Defaults to 0.6.
        dropout_pred (float, optional): Dropout rate of the layer before the prediction layer of the model.
                                        Defaults to 0.3.

        learning_rate (float, optional): Learning rate of the model in the training stage. Defaults to 0.0001.
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

    model_name: str = "image_model"
    image_base_model: str = "RestNet50"

    image_path: str = "./data/images/"
    log_path: str = "./logs/image_model/"
    model_path: str = "./model/image_model/weights/"
    transformed_image_path: str = "./data/adjusted_img/"

    batch_size = 32
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
        df_image_data = generator.generate_image_product_dataset()

        # encode the label
        le = preprocessing.LabelEncoder().fit(df_image_data["root_category"].unique())
        category = le.transform(df_image_data["root_category"].tolist())

        df_image_data['category'] = category

        self.classes = le.classes_
        self.num_class = len(self.classes)

        # split dataset
        df_train, df_val, df_test = generator.split_dataset(df_image_data)

        # to sync with milestone 5, if a product contains 2 or more images and
        # one of this is in training set, the others will also being put into
        # the training set. This is to avoid the data leaking problem, for instance,
        # the same product description is trained and tested in the text understanding model.
        df_train = pd.concat([df_train, df_val[df_val["product_id"].isin(df_train['product_id'].to_list())]])
        df_train = pd.concat([df_train, df_test[df_test["product_id"].isin(df_train['product_id'].to_list())]])

        df_val = df_val[~df_val["product_id"].isin(df_train['product_id'].to_list())]
        df_test = df_test[~df_test["product_id"].isin(df_train['product_id'].to_list())]

        y_train = df_train['category'].to_list()
        y_val = df_val['category'].to_list()
        y_test = df_test['category'].to_list()

        image_train = df_train['id_x'].to_list()
        image_val = df_val['id_x'].to_list()
        image_test = df_test['id_x'].to_list()

        # one hot encoded the category
        category_encoding_layer = tf.keras.layers.CategoryEncoding(
            num_tokens=self.num_class,
            output_mode="one_hot"
        )

        y_train = category_encoding_layer(y_train)
        y_val = category_encoding_layer(y_val)
        self.y_test = category_encoding_layer(y_test)

        gen = TFDatasetGenerator(
            images=image_train,
            tokens=None,
            num_max_tokens=-1,
            image_root_path=self.image_path,
            image_shape=self.image_shape,
            batch_size=self.batch_size,
            class_num=self.num_class,
            temp_img_path=self.transformed_image_path,
            labels=y_train
        )

        # This let tensorflow dataset know what is shape of dataset look like. We should output exactly the data same
        # shape in data generator to avoid any exception.
        out_sign = (
            tf.TensorSpec(shape=(None, self.image_shape[0], self.image_shape[1], self.image_shape[2]),
                          dtype=tf.float32),
            tf.TensorSpec(shape=(None, self.num_class), dtype=tf.float32)
        )

        ds_train = tf.data.Dataset.from_generator(gen, output_signature=out_sign)

        gen = TFDatasetGenerator(
            images=image_val,
            tokens=None,
            num_max_tokens=-1,
            image_root_path=self.image_path,
            image_shape=self.image_shape,
            batch_size=self.batch_size,
            class_num=self.num_class,
            temp_img_path=self.transformed_image_path,
            labels=y_val
        )

        ds_val = tf.data.Dataset.from_generator(gen, output_signature=out_sign)

        gen = TFDatasetGenerator(
            images=image_test,
            tokens=None,
            num_max_tokens=-1,
            image_root_path=self.image_path,
            image_shape=self.image_shape,
            batch_size=self.batch_size,
            class_num=self.num_class,
            temp_img_path=self.transformed_image_path,
            shuffle=False
        )

        ds_test = tf.data.Dataset.from_generator(gen, output_signature=out_sign)

        self.ds_train = ds_train.apply(
            tf.data.experimental.assert_cardinality(math.ceil(len(y_train) / self.batch_size)))
        self.ds_val = ds_val.apply(
            tf.data.experimental.assert_cardinality(math.ceil(len(y_val) / self.batch_size)))
        self.ds_test = ds_test.apply(
            tf.data.experimental.assert_cardinality(math.ceil(len(self.y_test) / self.batch_size)))

        return df_train, df_val, df_test

    def create_model(self) -> None:
        """
        Create an CNN model based on RestNet50V2 or EfficientNetV2. The model consists of preprocessing layer,
        data augmentation layer, global averaging layer, RestNet50V2 model, dropout layer and
        finally the prediction layer. The input shape can be configured in the class attributes
        input_shape and the output size will be equal to number of classes (determined in the
        prepare_data function).

        The model is compiled with AdamW optimiser together with learning rate scheduler. It takes advantages of
        decreasing learning rate as well as the adaptive learning rate for each parameter in each optimisation steps.
        It uses categorical cross entropy as loss function and accuracy
        as the evaluation metric.

        A compiled model will be saved in the model attributes as a result.

        This function will print out the summary of the model. You may also find the model graph and summary in README
        of this project.
        """

        inputs = tf.keras.layers.Input(shape=self.image_shape)

        img_augmentation = tf.keras.models.Sequential(
            [
                tf.keras.layers.RandomFlip('horizontal'),
                tf.keras.layers.RandomRotation(0.2)
            ],
            name="img_augmentation"
        )

        if self.image_base_model == "RestNet50":
            preprocess_input = tf.keras.applications.resnet_v2.preprocess_input
        elif self.image_base_model.startswith("EfficientNet"):
            preprocess_input = tf.keras.applications.efficientnet_v2.preprocess_input

        self.tf_image_base_model = TFImageTextUtil.prepare_image_base_model(
            self.image_base_model,
            self.image_shape
        )

        image_global_average_layer = tf.keras.layers.GlobalAveragePooling2D(name="pooling")
        image_dropout_0 = tf.keras.layers.Dropout(self.dropout_conv, name="dropout_0")
        image_dropout_1 = tf.keras.layers.Dropout(self.dropout_pred, name="dropout_1")
        image_dense_layer_0 = tf.keras.layers.Dense(1024, activation="relu", name="dense_0")
        image_dense_layer_1 = tf.keras.layers.Dense(256, activation="relu", name="dense_1")
        prediction = tf.keras.layers.Dense(self.num_class, name="prediction")

        layers = [
            self.tf_image_base_model,
            image_global_average_layer,
            image_dropout_0,
            image_dense_layer_0,
            image_dense_layer_1,
            image_dropout_1,
        ]

        self.image_seq_layers = tf.keras.Sequential(
            layers=layers, name="image_sequential"
        )

        x = img_augmentation(inputs)
        x = preprocess_input(x)
        x = self.image_seq_layers(x)
        outputs = prediction(x)

        steps_per_epoch = tf.data.experimental.cardinality(self.ds_train).numpy()
        num_train_steps = steps_per_epoch * self.epoch
        num_warmup_steps = int(0.1 * num_train_steps)

        optimizer = optimization.create_optimizer(init_lr=self.learning_rate,
                                                  num_train_steps=num_train_steps,
                                                  num_warmup_steps=num_warmup_steps,
                                                  optimizer_type='adamw')

        self.model = tf.keras.Model(inputs, outputs, name=self.model_name)
        self.model.compile(optimizer=optimizer,
                           loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

    def train_model(self) -> None:
        """
        Train a model with the training data. In each epoch, it will print out the loss and accuracy
        of the training and validation dataset in 'history' attribute. The records will be used for
        illustrating the performance of the model in later stage. There are 3 callbacks called early_stop_callback,
        tensorboard callback and hyperparameter callback: early_stop_callback will detect whether the model is overfitted
        and stop training while tensorboard callback and hyperparameter callback will create logs during the training
        process, and these logs can then be uploaded to TensorBoard.dev.

        """

        print("=" * 80)
        print("Start training")
        print("=" * 80)

        early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                               patience=5,
                                                               restore_best_weights=True)

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=self.log_path, histogram_freq=1)

        hparams_callback = hp.KerasCallback(self.log_path, {
            'dropout_conv': self.dropout_conv,
            'dropout_pred': self.dropout_pred
        })

        self.history = self.model.fit(self.ds_train,
                                      epochs=self.epoch,
                                      validation_data=self.ds_val,
                                      callbacks=[
                                          early_stop_callback,
                                          tensorboard_callback,
                                          hparams_callback
                                      ])

    def fine_tune_model(self) -> None:
        """
        Fine-tuning the model by unfreeze the image based model. Since the based model is usually pre-trained with a
        larger dataset, it will be better for us not to change the weights significantly, or we will lose the power of
        transfer learning. It uses the same components for training and validation. The result will be appended in to
        the history attribute.

        """
        if self.fine_tune_base_model:
            print("=" * 80)
            print("Start fine-tuning")
            print("=" * 80)

            TFImageTextUtil.set_base_model_trainable(self.tf_image_base_model,
                                                     self.fine_tune_base_model_layers)

            steps_per_epoch = tf.data.experimental.cardinality(self.ds_train).numpy()
            num_train_steps = steps_per_epoch * self.fine_tune_epoch
            num_warmup_steps = int(0.1 * num_train_steps)

            optimizer = optimization.create_optimizer(init_lr=self.fine_tune_learning_rate,
                                                      num_train_steps=num_train_steps,
                                                      num_warmup_steps=num_warmup_steps,
                                                      optimizer_type='adamw')

            self.model.compile(optimizer=optimizer,
                               loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                               metrics=['accuracy'])

            print(self.model.summary(expand_nested=True, show_trainable=True))

            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=self.log_path, histogram_freq=1)

            hparams_callback = hp.KerasCallback(self.log_path, {
                'dropout_conv': self.dropout_conv
            })

            early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                   patience=5,
                                                                   restore_best_weights=False)

            fine_tune_history = self.model.fit(self.ds_train,
                                               epochs=self.fine_tune_epoch,
                                               validation_data=self.ds_val,
                                               callbacks=[
                                                   early_stop_callback,
                                                   tensorboard_callback,
                                                   hparams_callback
                                               ])

            self.history.history['loss'].extend(fine_tune_history.history['loss'])
            self.history.history['val_loss'].extend(fine_tune_history.history['val_loss'])
            self.history.history['accuracy'].extend(fine_tune_history.history['accuracy'])
            self.history.history['val_accuracy'].extend(fine_tune_history.history['val_accuracy'])

    def evaluate_model(self) -> Tuple[float, float]:
        """
        Evaluate the model on the testing dataset. It returns only accuracy and loss to illustrate
        the overall performance of the model. If you need the prediction labels and the classification
        report returned, use predict_model instead.

        Returns:
            Tuple[float, float]: Loss and accuracy of the model.

        """

        prediction = self.model.predict(self.ds_test,
                                        batch_size=self.batch_size)

        y_true = [np.argmax(z) for z in self.y_test]
        y_pred = [np.argmax(x) for x in prediction]

        report = classification_report(y_true, y_pred, target_names=self.classes)
        print(report)

        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        loss = loss_fn(self.y_test, prediction)

        accuracy = sum([y_true[i] == y_pred[i] for i in range(len(y_true))]) / len(y_true)

        print(f"Evaluation on model accuracy {accuracy}, loss {loss}")

        return loss, accuracy

    def predict_model(self, dataset: Any) -> List[int]:
        """
        Predict with the model for records in a dataset.

        Args:
            dataset (Optional, Any): Dataset contains images.

        Returns:
            List[int]: List of labels.

        """

        prediction = self.model.predict(dataset)
        y_pred = [np.argmax(x) for x in prediction]

        return y_pred

    def save_model(self):
        """
        Save weight of the trained model.
        """

        super().save_model()
        self.image_seq_layers.save_weights(f"{self.model_path}img_seq_layers.ckpt")
