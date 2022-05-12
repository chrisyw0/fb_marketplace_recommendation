import pandas as pd
import tensorflow as tf
import numpy as np
from typing import Tuple, List, Any
from .cnn_model import CNNBaseModel
from .text_processing_util import TextUtil
from .image_text_tf_model import TFImageTextModel, TFImageModelDatasetGenerator
from ..data_preparation.prepare_dataset import DatasetGenerator
from tensorboard.plugins.hparams import api as hp
from dataclasses import field
from sklearn import preprocessing
from sklearn.metrics import classification_report


class ImageTextModel(CNNBaseModel):
    """
    A deep learning model predicting product category from its image, name and description.

    Args:
        df_image (pd.DataFrame): Image dataframe
        df_product (pd.DataFrame): Product dataframe

        model_name(str): Name of the model.
        embedding (str, Optional): The type of embedding model. Defaults to "Word2Vec".
        embedding_dim (int, Optional): The vector size of embedding model. Defaults to 300.
        embedding_pretrain_model (str, Optional): Whether to use a pretrain model to encode the text.
                                                  Defaults to None, which means no pretrained model is used.

        log_path (str, optional): Path to cache the training logs. Defaults to "./logs/text_model/".
        model_path (str, optional): Path to cache the weight of the image model. Defaults to "./model/text_model/weights/".
        transformed_image_path (str, optional): Path to cache the transformed image. This is improving when training,
                                                validating and testing the model as we don't need to transform and resize
                                                images when they are loaded into memory. Defaults to "./data/adjusted_img/"

        batch_size (int, optional): Batch size of the model. Defaults to 32.

        dropout_conv (float, optional): Dropout rate of the convolution layer of the model. Defaults to 0.6.
        dropout_prediction (float, optional): Dropout rate of the layer before the prediction layer of the model.
                                              Defaults to 0.4.
        learning_rate (float, optional): Learning rate of the model. Defaults to 0.01.
        input_shape (Tuple[int, int, int], Optional): Size of the image inputting to the model.
                                                      If image channel = 'RGB', the value will be
                                                      (width, height, 3) i.e. 3 channels
                                                      Defaults to (256, 256, 3)
        epoch (float, optional): Epoch of the model. Defaults to 50.
        metrics (List[str], optional):  list of metrics using for model evaluation. Defaults to ["accuracy"].

    """
    df_product: pd.DataFrame
    df_image: pd.DataFrame

    model_name = "image_text_model"

    log_path: str = "./logs/image_text_model/"
    model_path: str = "./model/image_text_model/weights/"
    image_path: str = "./data/images/"
    transformed_image_path: str = "./data/adjusted_img/"

    embedding: str = "Word2Vec"
    embedding_dim: int = 300
    embedding_pretrain_model: str = None

    batch_size: int = 32
    dropout_conv: float = 0.6
    dropout_prediction: float = 0.4
    learning_rate: float = 0.01

    image_shape: Tuple[int, int, int] = (256, 256, 3)

    epoch: int = 50
    metrics: List[str] = field(default_factory=lambda: ["accuracy"])

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare training, validation and testing data for the model. It includes building the word embedding model,
        splitting the dataset and getting essential elements for later stages.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Training, validation and testing dataframe

        """

        # merge the image and product dataset
        generator = DatasetGenerator(self.df_product, self.df_image)
        df_image_data = generator.generate_image_product_dataset()

        # split by . to make a list of sentences for each product
        product_sentences = df_image_data['product_name_description'].str.split('.').to_list()

        # whether to keep non word chars
        with_symbol = False if self.embedding == 'Word2Vec' else True

        print("Start tokenising the product name and description")

        # tokenise the sentences, and get training data for the embedding model
        # and get the maximum length in the product desciptions
        product_tokens, training_data, self.num_max_tokens = TextUtil.tokenise(
            product_sentences,
            self.embedding,
            with_symbol
        )

        print(f"Creating a {self.embedding} model, " \
              f"dimension {self.embedding_dim}, " \
              f"pre-train model {self.embedding_pretrain_model}")

        # create and train the embedding model
        self.embedding_model = TextUtil.prepare_embedding_model(
            embedding=self.embedding,
            embedding_dim=self.embedding_dim,
            training_data=training_data,
            pretrain_model=self.embedding_pretrain_model
        )

        print("Getting index from the embedding model")

        # convert token into token index in embedding model weight matrix
        df_image_data['tokens_index'] = TextUtil.get_token_index(
            product_tokens,
            self.embedding,
            self.embedding_model
        )

        print("Prepare training, validation and testing data")

        # encode the label
        le = preprocessing.LabelEncoder().fit(df_image_data["root_category"].unique())
        category = le.transform(df_image_data["root_category"].tolist())

        df_image_data['category'] = category

        self.classes = le.classes_
        self.num_class = len(self.classes)

        # split dataset
        df_train, df_val, df_test = generator.split_dataset(df_image_data)

        X_train = df_train['tokens_index'].to_list()
        X_val = df_val['tokens_index'].to_list()
        X_test = df_test['tokens_index'].to_list()

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

        gen = TFImageModelDatasetGenerator(image_train,
                                  X_train,
                                  self.num_max_tokens,
                                  self.image_path,
                                  self.image_shape,
                                  self.batch_size,
                                  self.num_class,
                                  self.transformed_image_path,
                                  y_train)

        # This let tensorflow dataset know what is shape of dataset look like. We should output exactly the data same
        # shape in data generator to avoid any exception.
        out_sign = (
            {
                "token": tf.TensorSpec(shape=(None, self.num_max_tokens), dtype=tf.int32),
                "image": tf.TensorSpec(shape=(None, self.image_shape[0], self.image_shape[1], self.image_shape[2]),
                                       dtype=tf.float32)},
            tf.TensorSpec(shape=(None, self.num_class), dtype=tf.float32)
        )

        self.ds_train = tf.data.Dataset.from_generator(gen, output_signature=out_sign)

        gen = TFImageModelDatasetGenerator(
            image_val,
            X_val,
            self.num_max_tokens,
            self.image_path,
            self.image_shape,
            self.batch_size,
            self.num_class,
            self.transformed_image_path,
            y_val
        )

        self.ds_val = tf.data.Dataset.from_generator(gen, output_signature=out_sign)

        gen = TFImageModelDatasetGenerator(
            image_test,
            X_test,
            self.num_max_tokens,
            self.image_path,
            self.image_shape,
            self.batch_size,
            self.num_class,
            self.transformed_image_path,
            shuffle=False
        )

        self.ds_test = tf.data.Dataset.from_generator(gen, output_signature=out_sign)

        return df_train, df_val, df_test

    def create_model(self) -> None:
        """
        It creates the model, compile and build the tensorflow model. Please check TFImageTextModel for more
        detail of the actual model.
        """
        embedding_layer = TextUtil.gensim_to_keras_embedding(
            self.embedding_model,
            train_embeddings=False,
            input_shape=(None, self.num_max_tokens))

        self.model = TFImageTextModel(
            self.num_class,
            embedding_layer,
            self.dropout_conv,
            self.dropout_prediction,
            self.image_shape
        )

        self.model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate / 10),
                           loss=tf.keras.losses.CategoricalCrossentropy(),
                           metrics=['accuracy'])

        self.model.build({
            "image": (self.batch_size, self.image_shape[0], self.image_shape[1], self.image_shape[2]),
            "token": (self.batch_size, self.num_max_tokens)
        })

        print("Model created")

    def train_model(self) -> None:
        """
        Train the model with the training data. It applies early stop by monitoring loss of validation dataset.
        If the model fails to improve for 8 epochs, it will stop training to avoid overfitting.
        In each epoch, it will print out the loss and accuracy of the training and validation dataset
        in 'history' attribute. The records will be used for illustrating the performance of the model
        in later stage. There are two callbacks called tensorboard callback and hyperparameter call back,
        it will create logs during the training process, and these logs can then be uploaded to TensorBoard.dev
        """

        print("Start training")

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=8,
                                                    restore_best_weights=True)

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=self.log_path,
            histogram_freq=1)

        hparams_callback = hp.KerasCallback(self.log_path, {
            'dropout_conv': self.dropout_conv,
            'dropout_prediction': self.dropout_prediction
        })

        self.history = self.model.fit(self.ds_train,
                                      batch_size=self.batch_size,
                                      epochs=self.epoch,
                                      validation_data=self.ds_val,
                                      callbacks=[
                                          callback,
                                          tensorboard_callback,
                                          hparams_callback
                                      ])

        print("Finish training")

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

        return loss, accuracy

    def predict_model(self, data: Any) -> List[int]:
        """
        Predict with the model for records in the dataset.

        Args:
            data: It should be a dataset created with TFImageModelDatasetGenerator. Please check prepare_data to
                  see how to create the dataset from the model.

        Returns:
            List[int]: List of labels

        """

        prediction = self.model.predict(data,
                                        batch_size=self.batch_size)

        y_pred = [np.argmax(x) for x in prediction]

        return y_pred
