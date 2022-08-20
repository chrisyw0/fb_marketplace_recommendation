import pandas as pd
import tensorflow as tf
import numpy as np

from typing import Tuple, List, Any
from dataclasses import field
from sklearn.metrics import classification_report

from fbRecommendation.dl.tensorflow.utils.tf_image_text_util import TFImageTextUtil
from fbRecommendation.dl.tensorflow.tf_base_classifier import TFBaseClassifier, get_optimizer
from fbRecommendation.dl.tensorflow.utils.tf_dataset_generator import TFDatasetGenerator
from fbRecommendation.dataPreparation.prepare_dataset import DatasetHelper


class TFTextClassifier(TFBaseClassifier):
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

    batch_size: int = 32
    dropout_conv: float = 0.5
    dropout_pred: float = 0.3
    learning_rate: float = 0.001
    epoch: int = 15

    fine_tune_base_model: bool = True
    fine_tune_learning_rate: float = 0.001
    fine_tune_epoch: int = 15

    metrics: List[str] = field(default_factory=lambda: ["accuracy"])

    def _get_model_name(self):
        return f"tf_text_model_{self.embedding}"

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

        # split by . to make a list of sentences for each product
        product_sentences = df_image_data['product_name_description'].to_list()

        # whether to keep non word chars
        with_symbol = False if self.embedding == 'Word2Vec' else True

        print("Start tokenising the product name and description")

        # tokenise the sentences, and get training data for the embedding model
        # and get the maximum length in the product descriptions
        product_tokens, training_data, self.num_max_tokens = TFImageTextUtil.tokenise(
            product_sentences,
            self.embedding,
            with_symbol
        )

        print(f"Creating a {self.embedding} model, " \
              f"dimension {self.embedding_dim}, " \
              f"pre-train model {self.embedding_pretrain_model}")

        # create and train the embedding model
        self.embedding_model = TFImageTextUtil.prepare_embedding_model(
            embedding=self.embedding,
            embedding_dim=self.embedding_dim,
            training_data=training_data,
            pretrain_model=self.embedding_pretrain_model
        )

        print("Getting index from the embedding model")

        # convert token into token index in embedding model weight matrix
        df_image_data['tokens_index'] = TFImageTextUtil.get_token_index(
            product_tokens,
            self.embedding,
            self.embedding_model
        )

        print("Prepare training, validation and testing data")

        self.num_class = len(self.classes)

        # split dataset
        df_train, df_val, df_test = generator.split_dataset(df_image_data)

        X_train = df_train['tokens_index'].to_list()
        X_val = df_val['tokens_index'].to_list()
        X_test = df_test['tokens_index'].to_list()

        y_train, y_val, y_test = generator.get_product_categories(df_train, df_val, df_test)

        # one hot encoded the category
        category_encoding_layer = tf.keras.layers.CategoryEncoding(
            num_tokens=self.num_class,
            output_mode="one_hot"
        )

        y_train = category_encoding_layer(y_train)
        y_val = category_encoding_layer(y_val)
        self.y_test = category_encoding_layer(y_test)

        gen = TFDatasetGenerator(images=None,
                                 tokens=X_train,
                                 num_max_tokens=self.num_max_tokens,
                                 image_root_path=None,
                                 image_shape=None,
                                 class_num=self.num_class,
                                 batch_size=self.batch_size,
                                 labels=y_train,
                                 pad_text_seq=True)

        # This let tensorflow dataset know what is shape of dataset look like. We should output exactly the data same
        # shape in data generator to avoid any exception.
        out_sign = (
            tf.TensorSpec(shape=(None, self.num_max_tokens), dtype=tf.int32),
            tf.TensorSpec(shape=(None, self.num_class), dtype=tf.float32)
        )

        self.ds_train = gen.get_dataset(out_sign)

        gen = TFDatasetGenerator(images=None,
                                 tokens=X_val,
                                 num_max_tokens=self.num_max_tokens,
                                 image_root_path=None,
                                 image_shape=None,
                                 class_num=self.num_class,
                                 batch_size=self.batch_size,
                                 labels=y_val,
                                 pad_text_seq=True)

        self.ds_val = gen.get_dataset(out_sign)

        gen = TFDatasetGenerator(images=None,
                                 tokens=X_test,
                                 num_max_tokens=self.num_max_tokens,
                                 image_root_path=None,
                                 image_shape=None,
                                 class_num=self.num_class,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 pad_text_seq=True)

        self.ds_test = gen.get_dataset(out_sign)

        return df_train, df_val, df_test

    def create_model(self) -> None:
        """
        Create the CNN model for text classification. It includes following layers:
        - Embedding layer: This layer is a layer with pre-built weights copied from embedding layer.
        - Conv1D: Convolution layer with activation layer applied
        - AveragePooling1D: Averaging pooling layer
        - Flatten: Flatten the 2D array input to 1D array
        - Dropout: Dropout a proportion of layers outputs from previous hidden layer
        - Dense: Linear layer with activation layer applied

        The input will be the token index for each product's tokens. To make the input shape the same for all products,
        padding is applied in previous function which simply append 0s to every product which has smaller tokens than
        the maximum, making the input shape to (batch_size, max number of tokens, 1).

        The output of the model will be the predicted probability of each class, which is equaled to
        (batch_size, num. of fbRecommendation)

        The model is compiled with AdamW optimiser together with learning rate scheduler. It takes advantages of
        decreasing learning rate as well as the adaptive learning rate for each parameter in each optimisation steps.
        It uses categorical cross entropy as loss function and accuracy as the evaluation metric.

        This function will print out the summary of the model. You may also find the model graph and summary in README
        of this project.

        """

        self.embedding_layer = TFImageTextUtil.gensim_to_keras_embedding(
            self.embedding_model,
            train_embeddings=False,
            input_shape=(self.num_max_tokens,))

        text_conv_layer_1 = tf.keras.layers.Conv1D(48, 3, activation="relu", name="text_conv_1")
        text_pooling_layer_1 = tf.keras.layers.AveragePooling1D(2, name="text_avg_pool_1")
        text_dropout_1 = tf.keras.layers.Dropout(self.dropout_conv, name="text_dropout_conv_1")
        text_conv_layer_2 = tf.keras.layers.Conv1D(24, 3, activation="relu", name="text_conv_2")
        text_pooling_layer_2 = tf.keras.layers.AveragePooling1D(2, name="text_avg_pool_2")
        text_flatten = tf.keras.layers.Flatten(name="text_flatten")
        text_dropout_2 = tf.keras.layers.Dropout(self.dropout_conv, name="text_dropout_conv_2")
        text_dense = tf.keras.layers.Dense(256, activation='relu', name="text_dense_1")
        text_dropout_pred = tf.keras.layers.Dropout(self.dropout_pred, name="text_dropout_pred_1")
        prediction = tf.keras.layers.Dense(self.num_class, name="prediction")

        inputs = tf.keras.layers.Input(shape=(self.num_max_tokens,), name="input")

        layers = [
            self.embedding_layer,
            text_conv_layer_1,
            text_pooling_layer_1,
            text_dropout_1,
            text_conv_layer_2,
            text_pooling_layer_2,
            text_flatten,
            text_dropout_2,
            text_dense,
            text_dropout_pred
        ]

        self.text_seq_layers = tf.keras.models.Sequential(layers=layers,
                                                          name="text_seq_layers")

        x = self.text_seq_layers(inputs)
        outputs = prediction(x)

        self.model = tf.keras.Model(inputs, outputs, name=self.model_name)

        optimizer = get_optimizer(
            self.ds_train,
            self.epoch,
            self.learning_rate
        )

        self.model.compile(optimizer=optimizer,
                           loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

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

        print("Start training")

        early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                               patience=5,
                                                               restore_best_weights=True)

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=self.log_path,
            histogram_freq=1,
            write_graph=False
        )

        self.history = self.model.fit(self.ds_train,
                                      batch_size=self.batch_size,
                                      epochs=self.epoch,
                                      validation_data=self.ds_val,
                                      callbacks=[
                                          early_stop_callback,
                                          tensorboard_callback
                                      ])

        print("Finish training")

    def fine_tune_model(self) -> None:
        """
        Fine-tuning the model by unfreeze the embedding. Our Word2Vec model is freeze in the training stage and
        not optimised for our final prediction. This fune-tuning stage will fine-tune the weights of all layers with
        lower learning rate to improve the performance of this model. It again applies early stop by monitoring loss of
        validation dataset.If the model fails to improve for 5 epochs, it will stop training to avoid overfitting.
        It uses the same components for training and validation. The result will be appended in to the
        history attribute.
        """

        if self.fine_tune_base_model:
            print("=" * 80)
            print("Start fine-tuning")
            print("=" * 80)

            TFImageTextUtil.set_base_model_trainable(self.embedding_layer, 1)

            optimizer = get_optimizer(
                self.ds_train,
                self.fine_tune_epoch,
                self.fine_tune_learning_rate
            )

            self.model.compile(optimizer=optimizer,
                               loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                               metrics=['accuracy'])

            print(self.model.summary(
                expand_nested=True,
                show_trainable=True
            ))

            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=self.log_path,
                histogram_freq=1,
                write_graph=False
            )

            # For some reason, the val accuracy keeps increasing even the val_loss increasing in last few epoch. This
            # may happen there are some outliner records that the model can't recognise, but the model actually improve
            # on other normal records.
            early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                                   patience=5,
                                                                   restore_best_weights=False)

            fine_tune_history = self.model.fit(self.ds_train,
                                               epochs=self.epoch+self.fine_tune_epoch,
                                               initial_epoch=self.epoch,
                                               validation_data=self.ds_val,
                                               callbacks=[
                                                   early_stop_callback,
                                                   tensorboard_callback
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

    def save_model(self):
        """
        Save weight of the trained model.
        """

        super().save_model()
        self.text_seq_layers.save_weights(f"{self.model_path}text_seq_layers.ckpt")

    def load_model(self):
        """
        Load the weight of the trained model.
        """

        super().load_model()
        self.text_seq_layers.load_weights(f"{self.model_path}text_seq_layers.ckpt").expect_partial()
