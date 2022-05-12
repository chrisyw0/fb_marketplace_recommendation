import numpy as np
import tensorflow as tf
import random
import math
import tempfile
import os
from pathlib import Path
from keras.preprocessing.sequence import pad_sequences
from typing import List, Tuple, Dict, Any, Generator


class TFImageModelDatasetGenerator:
    """
    This is the dataset generator fitting into Tensorflow dataset to control the output of the dataset.
    It also gets an option to shuffle the data when it completes a full load cycle.
    """
    def __init__(self, images: List[str], tokens: List[int], num_max_tokens: int,
                 image_root_path: str, image_shape: Tuple, batch_size: int,
                 class_num: int, temp_img_path: str = None,
                 labels: List[List[int]] = None, shuffle: bool = True) -> None:

        """
        Constructor of this dataset generator
        Args:
            images: List of image file names. Assuming all the images are stored in the same path (image_root_path).
            tokens: List of token index. This should be transformed according to the embedding model.
            num_max_tokens: The maximum number of tokens in the token index list.
            image_root_path: Image root path.
            image_shape: Image shape.
            batch_size: Batch size. It controls the number of records outputting in a batch.
            class_num: Number of output classes of the labels
            temp_img_path: A path to store the resized and transformed image. In this output generator, each image
                           will only be transformed and resized if the image is not found in this path and,
                           it will be saved into this path. After that, it will be loaded into memory for any
                           sub-sequence requests.
            labels: A list of labels of the desired class
            shuffle: Whether it should be shuffled or not when it completes a full load cycle
        """

        self.tokens = np.array(tokens)
        self.images = np.array(images)
        self.num_max_tokens = num_max_tokens
        self.image_shape = image_shape
        self.image_root_path = image_root_path
        self.labels = np.array(labels) if labels is not None else None
        self.batch_size = batch_size
        self.class_num = class_num
        self.shuffle = shuffle

        if self.shuffle:
            self.do_shuffle()

        self.temp_img_path = temp_img_path
        if self.temp_img_path is None:
            self.temp_img_path = tempfile.mkdtemp()
        else:
            os.makedirs(self.temp_img_path, exist_ok=True)

    def __len__(self) -> int:
        """
        Output the length of the dataset
        Returns:
            int: Length of the dataset.
        """
        return len(self.tokens)

    def __getitem__(self, idx) -> Tuple[Dict[str, Any], Any]:
        """
        Return a batch of data records. It loads image from original file path or from cache, token index with
        padding and labels of the predicted class in batch.
        Args:
            idx: batch index

        Returns:
            Tuple[Dict[str, Any], Any]: A batch of data records. It contains the image, token index and label in
                                        either numpy array or Tensorflow tensor format. Tensorflow will transform
                                        it into corresponding tensor if required.

        """
        # print(f"get item at index {idx}")

        def _get_image_tensor(file_path: str, cache_path: str):
            """
            Get the transformed image tensor from file path or from cache. If the transformed is found in cached
            path, it will load it from there or otherwise load from the original file from file path, transformed it
            and save it in the cache path.
            Args:
                file_path: The original file path. This should not be a relative path or tf.io.read_file may not be
                           able to load it.
                cache_path: The cache file path. This should not be a relative path or tf.io.read_file may not be
                            able to load it.

            Returns:
                The tensor of the image.

            """
            if os.path.exists(cache_path):
                img = tf.io.read_file(cache_path)
                img = tf.io.decode_image(img, channels=3)

                return img

            # print(file_path)
            img = tf.io.read_file(file_path)
            img = tf.io.decode_image(img, channels=3)
            img = tf.image.resize_with_pad(img, self.image_shape[0], self.image_shape[1])

            save_img = tf.cast(img, tf.uint8)
            save_img = tf.io.encode_jpeg(save_img, format='rgb')
            tf.io.write_file(tf.constant(cache_path), save_img)

            return img

        images = []
        text = pad_sequences(self.tokens[list(idx)], maxlen=self.num_max_tokens, padding="post")

        for i in idx:
            image_path = self.images[i]
            img = _get_image_tensor(str(Path(f"{self.image_root_path}{image_path}.jpg").resolve()),
                                    str(Path(f"{self.temp_img_path}{image_path}.jpg").resolve()))

            images.append(img)

        # print(images, text)

        if self.labels is not None:
            return {"token": text, "image": images}, self.labels[idx]
        else:
            return {"token": text, "image": images}, np.zeros((self.batch_size, self.class_num))

    def __call__(self) -> Generator[Tuple[Dict[str, Any], Any], None, None]:
        """
        This is the entry point when tensorflow try to get a batch of data and generate (yield) a batch of
        records at a time.
        Returns:
            Generator[Tuple[Dict[str, Any], Any]]: A generator with decoration same as __getitem__()

        """
        num_of_batch = math.ceil(self.__len__() / self.batch_size)
        for i in range(num_of_batch):
            idx = range(i * self.batch_size, min(self.__len__(), (i + 1) * self.batch_size))
            yield self.__getitem__(idx)

            if self.shuffle and i == num_of_batch - 1:
                self.do_shuffle()

    def do_shuffle(self):
        """
        Shuffle the data in the dataset.
        """
        reidx = random.sample(population=list(range(self.__len__())), k=self.__len__())

        self.images = self.images[reidx]
        self.tokens = self.tokens[reidx]

        if self.labels is not None:
            self.labels = self.labels[reidx]


class TFImageTextModel(tf.keras.Model):
    """
    This is the actual classification model (inherent to tensorflow keras model) consisting all the layers
    and baseline models. It is actually combining Text and Image model, the image data and text data
    will go through it corresponding CNN layer and base model (RestNetV50), and finally combining
    the final dense layer and output the probability of each class by a final softmax prediction layer.
    You may find more detail in ImageModel and TextModel.
    """
    def __init__(self,
                 output_size: int,
                 text_embedding_layer: Any,
                 dropout_conv: float,
                 dropout_prediction: float,
                 image_shape: Tuple):

        """
        Constructor of the model. This setup all the layers and base line model for this model.
        Args:
            output_size: The number of class of the final output layer.
            text_embedding_layer: The pre-trained word embedding layer.
            dropout_conv: Dropout rate of the convolution layer
            dropout_prediction: Dropout rate of the prediction layer.
            image_shape: The image shape input to the RestNetV50 model
        """

        super(TFImageTextModel, self).__init__()

        self.image_shape = image_shape

        self.text_embedding_layer = text_embedding_layer

        self.text_conv_layer_1 = tf.keras.layers.Conv1D(48, 3, activation="relu")
        self.text_pooling_layer_1 = tf.keras.layers.AveragePooling1D(2)
        self.text_dropout_1 = tf.keras.layers.Dropout(dropout_conv)
        self.text_conv_layer_2 = tf.keras.layers.Conv1D(24, 3, activation="relu")
        self.text_pooling_layer_2 = tf.keras.layers.AveragePooling1D(2)
        self.text_flatten = tf.keras.layers.Flatten()
        self.text_dropout_2 = tf.keras.layers.Dropout(dropout_conv)
        self.text_dense = tf.keras.layers.Dense(128, activation='relu')
        self.text_dropout_pred = tf.keras.layers.Dropout(dropout_prediction)

        self.image_preprocess_input = tf.keras.applications.resnet_v2.preprocess_input

        self.image_data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip('horizontal'),
            tf.keras.layers.RandomRotation(0.2),
        ])

        self.image_base_model = tf.keras.applications.ResNet50V2(include_top=False,
                                                                 input_shape=image_shape)
        self.image_base_model.trainable = False

        self.image_global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        self.image_dropout = tf.keras.layers.Dropout(dropout_conv)

        self.image_dense_layer = tf.keras.layers.Dense(128, activation="relu")
        self.image_dropout_pred = tf.keras.layers.Dropout(dropout_prediction)

        self.prediction_layer = tf.keras.layers.Dense(output_size, activation="softmax")

    def call(self, inputs, training=True, **kwargs):
        """
        This is where the actual data comes in and determines how it goes through the layers.
        Args:
            inputs: A batch of input data.
            training: Whether it is training or testing.
            **kwargs: Some other keyword arguments

        Returns:
            Final prediction result

        """
        text_data = inputs["token"]
        image_data = inputs["image"]

        # print(f"Text data input shape {text_data.shape}")

        text_data = self.text_embedding_layer(text_data)

        text_data = self.text_conv_layer_1(text_data)
        text_data = self.text_pooling_layer_1(text_data)
        text_data = self.text_dropout_1(text_data)
        text_data = self.text_conv_layer_2(text_data)
        text_data = self.text_pooling_layer_2(text_data)
        text_data = self.text_dropout_2(text_data)
        text_data = self.text_flatten(text_data)
        text_data = self.text_dense(text_data)
        text_data = self.text_dropout_pred(text_data)

        # print(f"finishing text, text shape {text_data.shape}")

        # print(f"Image data read, shape = {image_data.shape}")

        image_data = self.image_preprocess_input(image_data)
        image_data = self.image_data_augmentation(image_data)
        image_data = self.image_base_model(image_data)
        image_data = self.image_global_average_layer(image_data)
        image_data = self.image_dropout(image_data)
        image_data = self.image_dense_layer(image_data)
        image_data = self.image_dropout_pred(image_data)

        # print(f"finishing image, image shape {image_data.shape}")

        x = tf.concat([text_data, image_data], 1)

        # print(f"Combine image and text, shape {x.shape}")
        x = self.prediction_layer(x)

        return x
