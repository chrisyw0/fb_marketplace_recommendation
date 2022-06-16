import tensorflow as tf
import numpy as np
import random
import math
import tempfile
import os

from typing import List, Tuple, Dict, Any, Generator, Optional
from pathlib import Path
from keras.preprocessing.sequence import pad_sequences


class TFDatasetGenerator:
    """
    This is the dataset generator fitting into Tensorflow dataset to control the output of the dataset.
    It also gets an option to shuffle the data when it completes a full load cycle.
    """

    def __init__(self, images: Optional[List[str]], tokens: Optional[List[int]], num_max_tokens: int,
                 image_root_path: Optional[str], image_shape: Optional[Tuple], batch_size: int,
                 class_num: int, temp_img_path: Optional[str] = None,
                 labels: Optional[List[List[int]]] = None, shuffle: bool = True,
                 pad_text_seq: bool = False) -> None:

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
            pad_text_seq: Whether it should pad 0 if the input have various length
        """

        self.tokens = np.array(tokens) if tokens is not None else None
        self.images = np.array(images) if images is not None else None
        self.num_max_tokens = num_max_tokens
        self.image_shape = image_shape
        self.image_root_path = image_root_path
        self.labels = np.array(labels) if labels is not None else None
        self.batch_size = batch_size
        self.class_num = class_num
        self.shuffle = shuffle
        self.pad_text_seq = pad_text_seq

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

        if self.tokens is not None:
            return len(self.tokens)
        else:
            return len(self.images)

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
                _img = tf.io.read_file(cache_path)
                _img = tf.io.decode_image(_img, channels=3)
                return _img

            # print(file_path)
            _img = tf.io.read_file(file_path)
            _img = tf.io.decode_image(_img, channels=3)
            _img = tf.image.resize_with_pad(_img,
                                            self.image_shape[0],
                                            self.image_shape[1])

            save_img = tf.cast(_img, tf.uint8)
            save_img = tf.io.encode_jpeg(save_img, format='rgb')
            tf.io.write_file(tf.constant(cache_path), save_img)

            return _img

        if self.tokens is not None:
            text = self.tokens[list(idx)]

            if self.pad_text_seq:
                text = pad_sequences(text, maxlen=self.num_max_tokens, padding="post")
        else:
            text = None

        if self.images is not None:
            images = []
            for i in idx:
                image_path = self.images[i]
                img = _get_image_tensor(str(Path(f"{self.image_root_path}{image_path}.jpg").resolve()),
                                        str(Path(f"{self.temp_img_path}{image_path}.jpg").resolve()))

                images.append(img)
        else:
            images = None

        # print(images, text)

        result_labels = self.labels[idx] if self.labels is not None else \
            np.zeros((self.batch_size, self.class_num))

        if text is not None and images is not None:
            return {"token": text, "image": images}, result_labels
        elif text is not None:
            return text, result_labels
        else:
            return images, result_labels

    def __call__(self) -> Generator[Tuple[Dict[str, Any], Any], None, None]:
        """
        This is the entry point when tensorflow try to get a batch of data and generate (yield) a batch of
        records at a time.
        Returns:
            Generator[Tuple[Dict[str, Any], Any]]: A generator with decoration same as __getitem__()

        """
        num_of_batch = math.ceil(self.__len__() / self.batch_size)
        for i in range(num_of_batch):
            # print(f"getting next batch: {i}, {self.batch_size}, {self.__len__()}")
            idx = range(i * self.batch_size, min(self.__len__(), (i + 1) * self.batch_size))
            yield self.__getitem__(idx)

            if self.shuffle and i == num_of_batch - 1:
                self.do_shuffle()

    def do_shuffle(self):
        """
        Shuffle the data in the dataset.
        """
        reidx = random.sample(population=list(range(self.__len__())), k=self.__len__())

        if self.images is not None:
            self.images = self.images[reidx]

        if self.tokens is not None:
            self.tokens = self.tokens[reidx]

        if self.labels is not None:
            self.labels = self.labels[reidx]