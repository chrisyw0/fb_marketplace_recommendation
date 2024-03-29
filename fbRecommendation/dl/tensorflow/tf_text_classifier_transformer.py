import pandas as pd
import tensorflow as tf
import numpy as np

from typing import Tuple, List, Any
from dataclasses import field
from sklearn.metrics import classification_report

from fbRecommendation.dl.tensorflow.utils.tf_image_text_util import TFImageTextUtil
from fbRecommendation.dl.tensorflow.model.tf_model import TFTextTransformerModel
from fbRecommendation.dl.tensorflow.tf_base_classifier import TFBaseClassifier, get_optimizer
from fbRecommendation.dl.tensorflow.utils.tf_dataset_generator import TFDatasetGenerator
from fbRecommendation.dataset.prepare_dataset import DatasetHelper


class TFTextTransformerClassifier(TFBaseClassifier):
    """
    A deep learning model predicting product category from its name and description.

    Args:
        df_image (pd.DataFrame): Image dataframe
        df_product (pd.DataFrame): Product dataframe

        embedding (str, Optional): The type of embedding model. Defaults to "BERT".

        embedding_dim (int, Optional): The vector size of embedding model. Defaults to 768.
        embedding_pretrain_model (str, Optional): Whether to use a pretrain model to encode the text. Please check
                                                  tfhub_text_model_constant.py for available options.
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
    embedding_pretrain_model: str = "bert_en_cased_L-12_H-768_A-12"

    batch_size: int = 16
    dropout_pred: float = 0.5
    epoch: int = 5
    learning_rate: float = 2e-5

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

        print("Prepare training, validation and testing data")

        df_image_data['product_name_description'] = df_image_data["product_name_description"].apply(
            TFImageTextUtil.clean_text)

        self.num_class = len(self.classes)

        # split dataset
        df_train, df_val, df_test = generator.split_dataset(df_image_data)

        X_train = df_train['product_name_description'].to_list()
        X_val = df_val['product_name_description'].to_list()
        X_test = df_test['product_name_description'].to_list()

        y_train, y_val, y_test = generator.get_product_categories(df_train, df_val, df_test)

        # one hot encoded the category
        category_encoding_layer = tf.keras.layers.CategoryEncoding(
            num_tokens=self.num_class,
            output_mode="one_hot"
        )

        y_train = category_encoding_layer(y_train)
        y_val = category_encoding_layer(y_val)
        self.y_test = category_encoding_layer(y_test)

        gen = TFDatasetGenerator(
            images=None,
            tokens=X_train,
            num_max_tokens=-1,
            image_root_path=None,
            image_shape=None,
            class_num=self.num_class,
            batch_size=self.batch_size,
            labels=y_train
        )

        # This let tensorflow dataset know what is shape of dataset look like. We should output exactly the data same
        # shape in data generator to avoid any exception.
        self.out_sign = (
            tf.TensorSpec(shape=(None,), dtype=tf.string),
            tf.TensorSpec(shape=(None, self.num_class), dtype=tf.float32)
        )

        self.ds_train = gen.get_dataset(self.out_sign)

        gen = TFDatasetGenerator(
            images=None,
            tokens=X_val,
            num_max_tokens=-1,
            image_root_path=None,
            image_shape=None,
            class_num=self.num_class,
            batch_size=self.batch_size,
            labels=y_val
        )

        self.ds_val = gen.get_dataset(self.out_sign)

        gen = TFDatasetGenerator(
            images=None,
            tokens=X_test,
            num_max_tokens=-1,
            image_root_path=None,
            image_shape=None,
            class_num=self.num_class,
            batch_size=self.batch_size,
            shuffle=False
        )

        self.ds_test = gen.get_dataset(self.out_sign)

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
        (batch_size, num. of fbRecommendation)

        The model is compiled with AdamW optimiser together with learning rate scheduler. It takes advantages of
        decreasing learning rate as well as the adaptive learning rate for each parameter in each optimisation steps.
        It uses categorical cross entropy as loss function and accuracy as the evaluation metric.

        This function will print out the summary of the model. You may also find the model graph and summary in README
        of this project.
        """

        optimizer = get_optimizer(
            self.ds_train,
            self.epoch,
            self.learning_rate
        )

        kwargs = {
            "num_class": self.num_class,
            "model_name": self.model_name,
            "embedding": self.embedding,
            "embedding_dim": self.embedding_dim,
            "embedding_pretrain_model": self.embedding_pretrain_model,
            "dropout_pred": self.dropout_pred
        }

        self.model, self.text_seq_layer, self.embedding_model = TFTextTransformerModel.get_model(**kwargs)

        self.model.compile(optimizer=optimizer,
                           loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

        print("Model created")

    def train_model(self) -> None:
        """
        Train the model with the training data. It applies early stop by monitoring loss of validation dataset.
        If the model fails to improve for 2 epochs, it will stop training to avoid overfitting.
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

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=2,
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
                                          callback,
                                          tensorboard_callback
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

        # print(prediction)

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
