import pandas as pd
import tensorflow as tf
import numpy as np
from typing import Tuple, List, Any
from .text_processing_util import TextUtil
from .cnn_model import CNNBaseModel
from ..data_preparation.prepare_dataset import DatasetGenerator
from tensorboard.plugins.hparams import api as hp

from dataclasses import field
from keras.preprocessing.sequence import pad_sequences
from sklearn import preprocessing
from sklearn.metrics import classification_report


class TextModel(CNNBaseModel):
    """
    A deep learning model predicting product category from its name and description.

    Args:
        df_image (pd.DataFrame): Image dataframe
        df_product (pd.DataFrame): Product dataframe

        model_name(str): Name of the model.

        log_path (str, optional): Path to cache the training logs. Defaults to "./logs/text_model/".
        model_path (str, optional): Path to cache the weight of the image model. Defaults to "./model/text_model/weights/".

        embedding (str, Optional): The type of embedding model. Defaults to "Word2Vec".
        embedding_dim (int, Optional): The vector size of embedding model. Defaults to 300.
        embedding_pretrain_model (str, Optional): Whether to use a pretrain model to encode the text.
                                                  Defaults to None, which means no pretrained model is used.

        batch_size (int, optional): Batch size of the model. Defaults to 32.

        dropout_conv (float, optional): Dropout rate of the convolution layer of the model. Defaults to 0.5.
        dropout_prediction (float, optional): Dropout rate of the layer before the prediction layer of the model.
                                              Defaults to 0.3.
        learning_rate (float, optional): Learning rate of the model. Defaults to 0.01.

        epoch (float, optional): Epoch of the model. Defaults to 50.
        metrics (List[str], optional):  list of metrics using for model evaluation. Defaults to ["accuracy"].

    """
    df_product: pd.DataFrame
    df_image: pd.DataFrame

    model_name = "text_model"

    log_path: str = "./logs/text_model/"
    model_path: str = "./model/text_model/weights/"

    embedding: str = "Word2Vec"
    embedding_dim: int = 300
    embedding_pretrain_model: str = None

    batch_size: int = 32
    dropout_conv: float = 0.5
    dropout_prediction: float = 0.3
    learning_rate: float = 0.01

    epoch: int = 50
    metrics: List[str] = field(default_factory=lambda: ["accuracy"])

    def __init__(self, df_image: pd.DataFrame, df_product: pd.DataFrame):
        self.df_image = df_image
        self.df_product = df_product

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

        # since we merge image with product dataframe, and some products have more than one
        # images, we don't want the information leak from training dataset into validation and
        # testing dataset, we should remove those from testing and validation datasets
        df_val = df_val[~df_val["product_id"].isin(df_train['product_id'].to_list())]
        df_test = df_test[~df_test["product_id"].isin(df_train['product_id'].to_list())]

        X_train = df_train['tokens_index'].to_list()
        X_val = df_val['tokens_index'].to_list()
        X_test = df_test['tokens_index'].to_list()

        y_train = df_train['category']
        y_val = df_val['category']
        y_test = df_test['category']

        # pad the tokens index list for each product to make all data have the same length
        self.X_train = pad_sequences(X_train, maxlen=self.num_max_tokens, padding="post")
        self.X_val = pad_sequences(X_val, maxlen=self.num_max_tokens, padding="post")
        self.X_test = pad_sequences(X_test, maxlen=self.num_max_tokens, padding="post")

        # one hot encoded the category
        category_encoding_layer = tf.keras.layers.CategoryEncoding(
            num_tokens=self.num_class,
            output_mode="one_hot"
        )

        self.y_train = category_encoding_layer(y_train)
        self.y_val = category_encoding_layer(y_val)
        self.y_test = category_encoding_layer(y_test)

        print(f"Finish preparing data, shape of X_train {self.X_train.shape}, " \
              f"shape of X_val {self.X_val.shape}, " \
              f"shape of X_test {self.X_test.shape}, " \
              f"shape of y_train {self.y_train.shape}, " \
              f"shape of y_val {self.y_val.shape}, " \
              f"shape of y_test {self.y_test.shape}")

        return df_train, df_val, df_test

    def create_model(self) -> None:
        """
        Create the CNN model for text classification. It includes following layers:
        - Embedding layer: This layer is a non-trainable layer with pre-built weights copied from embedding layer.
        - Conv1D: Convolution layer with activation layer applied
        - AveragePooling1D: Averaging pooling layer
        - Flatten: Flatten the 2D array input to 1D array
        - Dropout: Dropout a proportion of layers outputs from previous hidden layer
        - Dense: Linear layer with activation layer applied

        The input will be the token index for each product's tokens. To make the input shape the same for all products,
        padding is applied in previous function which simply append 0s to every product which has smaller tokens than
        the maximum, making the input shape to (batch_size, max number of tokens, 1).

        The output of the model will be the predicted probability of each class, which is equaled to
        (batch_size, num. of classes)

        This function will print out the summary of the model. Here is an example.

        Model: "sequential"
        _________________________________________________________________
         Layer (type)                Output Shape              Param #
        =================================================================
         embedding (Embedding)       (None, 1458, 300)         8397600

         conv1d (Conv1D)             (None, 1456, 48)          43248

         average_pooling1d (AverageP  (None, 728, 48)          0
         ooling1D)

         dropout (Dropout)           (None, 728, 48)           0

         conv1d_1 (Conv1D)           (None, 726, 24)           3480

         average_pooling1d_1 (Averag  (None, 363, 24)          0
         ePooling1D)

         flatten (Flatten)           (None, 8712)              0

         dropout_1 (Dropout)         (None, 8712)              0

         dense (Dense)               (None, 256)               2230528

         dropout_2 (Dropout)         (None, 256)               0

         dense_1 (Dense)             (None, 13)                3341

        =================================================================
        Total params: 10,678,197
        Trainable params: 2,280,597
        Non-trainable params: 8,397,600
        _________________________________________________________________

        The model will finally be compiled with RMSprop optimiser, categorical cross-entropy loss and
        accuracy as metrics. It will be saved as class attributes for later use.

        """
        embedding_layer = TextUtil.gensim_to_keras_embedding(
            self.embedding_model,
            train_embeddings=False,
            input_shape=(self.num_max_tokens,))

        self.model = tf.keras.Sequential([
            embedding_layer,
            tf.keras.layers.Conv1D(48, 3, activation="relu"),
            tf.keras.layers.AveragePooling1D(2),
            tf.keras.layers.Dropout(self.dropout_conv),
            tf.keras.layers.Conv1D(24, 3, activation="relu"),
            tf.keras.layers.AveragePooling1D(2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(self.dropout_conv),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(self.dropout_prediction),
            tf.keras.layers.Dense(self.num_class, activation="softmax")
        ])

        self.model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate / 10),
                           loss=tf.keras.losses.CategoricalCrossentropy(),
                           metrics=['accuracy'])

        print("Model created")

    def train_model(self) -> None:
        """
        Train the model with the training data. It applies early stop by monitoring loss of validation dataset.
        If the model fails to improve for 8 epochs, it will stop training to avoid overfitting.
        In each epoch, it will print out the loss and accuracy of the training and validation dataset
        in 'history' attribute. The records will be used for illustrating the performance of the model
        in later stage. There are two callbacks called tensorboard callback and hyperparameter call back,
        it will create logs during the training process, and these logs can then be uploaded to TensorBoard.dev

        Returns:

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

        self.history = self.model.fit(self.X_train,
                                      self.y_train,
                                      batch_size=self.batch_size,
                                      epochs=self.epoch,
                                      validation_data=(self.X_val, self.y_val),
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

        prediction = self.model.predict(self.X_test,
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
            data:

        Returns:
            List[int]: List of labels

        """

        prediction = self.model.predict(data,
                                        batch_size=self.batch_size)

        y_pred = [np.argmax(x) for x in prediction]

        return y_pred
