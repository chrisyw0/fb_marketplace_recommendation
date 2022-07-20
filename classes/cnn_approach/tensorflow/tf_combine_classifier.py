import pandas as pd
import tensorflow as tf
import numpy as np
import math

from typing import Tuple, List, Any, Optional
from dataclasses import field
from sklearn.metrics import classification_report
from official.nlp import optimization

from classes.data_preparation.prepare_dataset import DatasetHelper
from classes.cnn_approach.tensorflow.tf_base_classifier import TFBaseClassifier
from classes.cnn_approach.tensorflow.utils.tf_image_text_util import TFImageTextUtil
from classes.cnn_approach.tensorflow.utils.tf_dataset_generator import TFDatasetGenerator


class TFImageTextClassifier(TFBaseClassifier):
    """
    A deep learning model predicting product category from its image, name and description.

    Args:
        df_image (pd.DataFrame): Image dataframe
        df_product (pd.DataFrame): Product dataframe

        image_seq_layers (tf.keras.Model): The trained image model (without preprocessing and prediction layer).
        text_seq_layers (tf.keras.Model): The trained text model (without preprocessing and prediction layer).

        image_base_model (str, optional): Name of the image pre-trained model. Defaults to "EfficientNetB3"
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

    image_base_model: str = "EfficientNetB3"
    embedding: str = "Word2Vec"

    image_path: str = "./data/images/"
    transformed_image_path: str = "./data/adjusted_img/"

    embedding_pretrain_model: str = None

    image_shape: Tuple[int, int, int] = (300, 300, 3)
    transformed_image_path += str(image_shape[0]) + "/"

    epoch: int = 3
    learning_rate: float = 1e-3
    batch_size: int = 16

    metrics: List[str] = field(default_factory=lambda: ["accuracy"])

    def __init__(self,
                 df_image: pd.DataFrame,
                 df_product: pd.DataFrame,
                 image_seq_layers: Any,
                 text_seq_layers: Any,
                 embedding_model: Optional[Any]
                 ):

        super().__init__(df_image, df_product)

        self.image_seq_layers = tf.keras.models.clone_model(image_seq_layers)
        self.image_seq_layers.set_weights(image_seq_layers.get_weights())

        self.text_seq_layers = tf.keras.models.clone_model(text_seq_layers)
        self.text_seq_layers.set_weights(text_seq_layers.get_weights())

        self.is_transformer_based_text_model = self.text_seq_layers.name == 'text_seq_layer_transformer'

        self.embedding_model = embedding_model

    def _get_model_name(self):
        return f"tf_image_text_model_{self.image_base_model}_{self.embedding}"

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

        if not self.is_transformer_based_text_model:
            token_input_key = 'tokens_index'

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

            # convert token into token index in embedding model weight matrix
            df_image_data[token_input_key] = TFImageTextUtil.get_token_index(
                product_tokens,
                self.embedding,
                self.embedding_model
            )

            token_tensor_spec = tf.TensorSpec(shape=(None, self.num_max_tokens), dtype=tf.int32)

        else:
            token_input_key = 'product_name_description'
            token_tensor_spec = tf.TensorSpec(shape=(None,), dtype=tf.string)

            df_image_data['product_name_description'] = df_image_data["product_name_description"].apply(
                TFImageTextUtil.clean_text)

            self.num_max_tokens = -1

        print("Prepare training, validation and testing data")

        self.num_class = len(self.classes)

        # split dataset
        df_train, df_val, df_test = generator.split_dataset(df_image_data)

        X_train = df_train[token_input_key].to_list()
        X_val = df_val[token_input_key].to_list()
        X_test = df_test[token_input_key].to_list()

        y_train, y_val, y_test = generator.get_product_categories(df_train, df_val, df_test)

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
            tokens=X_train,
            num_max_tokens=self.num_max_tokens,
            image_root_path=self.image_path,
            image_shape=self.image_shape,
            batch_size=self.batch_size,
            class_num=self.num_class,
            temp_img_path=self.transformed_image_path,
            labels=y_train,
            pad_text_seq= not self.is_transformer_based_text_model
        )

        # This let tensorflow dataset know what is shape of dataset look like. We should output exactly the data same
        # shape in data generator to avoid any exception.
        out_sign = (
            {
                "token": token_tensor_spec,
                "image": tf.TensorSpec(shape=(None, self.image_shape[0], self.image_shape[1], self.image_shape[2]),
                                       dtype=tf.float32)},
            tf.TensorSpec(shape=(None, self.num_class), dtype=tf.float32)
        )

        ds_train = tf.data.Dataset.from_generator(gen, output_signature=out_sign)

        gen = TFDatasetGenerator(
            images=image_val,
            tokens=X_val,
            num_max_tokens=self.num_max_tokens,
            image_root_path=self.image_path,
            image_shape=self.image_shape,
            batch_size=self.batch_size,
            class_num=self.num_class,
            temp_img_path=self.transformed_image_path,
            labels=y_val,
            pad_text_seq= not self.is_transformer_based_text_model
        )

        ds_val = tf.data.Dataset.from_generator(gen, output_signature=out_sign)

        gen = TFDatasetGenerator(
            images=image_test,
            tokens=X_test,
            num_max_tokens=self.num_max_tokens,
            image_root_path=self.image_path,
            image_shape=self.image_shape,
            batch_size=self.batch_size,
            class_num=self.num_class,
            temp_img_path=self.transformed_image_path,
            shuffle=False,
            pad_text_seq= not self.is_transformer_based_text_model
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
        self.image_seq_layers.trainable = False
        self.text_seq_layers.trainable = False

        img_augmentation = tf.keras.models.Sequential(
            [
                tf.keras.layers.RandomFlip('horizontal'),
                tf.keras.layers.RandomRotation(0.2)
            ],
            name="img_augmentation"
        )

        prediction = tf.keras.layers.Dense(self.num_class, name="prediction")

        if self.is_transformer_based_text_model:
            text_inputs = tf.keras.layers.Input(shape=(), dtype=tf.string, name="input")
        else:
            text_inputs = tf.keras.layers.Input(shape=(self.num_max_tokens,), dtype=tf.int32, name="input")

        image_inputs = tf.keras.layers.Input(shape=self.image_shape, dtype=tf.float32)

        x_text = self.text_seq_layers(text_inputs)
        x_img = img_augmentation(image_inputs)

        if self.image_base_model == "RestNet50":
            preprocess_input = tf.keras.applications.resnet_v2.preprocess_input
            x_img = preprocess_input(x_img)
        elif self.image_base_model.startswith("EfficientNet"):
            preprocess_input = tf.keras.applications.efficientnet_v2.preprocess_input
            x_img = preprocess_input(x_img)

        x_img = self.image_seq_layers(x_img)
        x = tf.concat([x_text, x_img], 1)

        outputs = prediction(x)
        self.model = tf.keras.Model(
            {
                "token": text_inputs,
                "image": image_inputs
            },
            outputs, name=self.model_name
        )

        steps_per_epoch = tf.data.experimental.cardinality(self.ds_train).numpy()
        num_train_steps = steps_per_epoch * self.epoch
        num_warmup_steps = int(0.1 * num_train_steps)

        optimizer = optimization.create_optimizer(init_lr=self.learning_rate,
                                                  num_train_steps=num_train_steps,
                                                  num_warmup_steps=num_warmup_steps,
                                                  optimizer_type='adamw')

        self.model.compile(optimizer=optimizer,
                           loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

        print("Model created")

    def train_model(self) -> None:
        """
        Train the model with the training data. It applies early stop by monitoring loss of validation dataset.
        In each epoch, it will print out the loss and accuracy of the training and validation dataset
        in 'history' attribute. The records will be used for illustrating the performance of the model
        in later stage. There is a callback called tensorboard callback, which creates logs during the training process,
        and these logs can then be uploaded to TensorBoard.dev
        """

        print("Start training")

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
