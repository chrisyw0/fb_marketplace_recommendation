import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import keras

from typing import List, Tuple, Any
from .tf_text_processing_constant import TFTextProcessingConstant
from fbRecommendation.dl.base.image_text_util import ImageTextUtil


class TFImageTextUtil(ImageTextUtil):
    """
    This class is a subclass ImageTextUtil, provide utils functions for tensorflow model
    """
    @staticmethod
    def prepare_embedding_model(embedding: str,
                                embedding_dim: int,
                                training_data: List[List[str]] = None,
                                window: int = 2,
                                min_count: int = 1,
                                pretrain_model: str = None,
                                trainable: bool = True) -> Any:

        """
        Create a word embedding model.
        Args:
            embedding: Type of embedding to be used for the model
            embedding_dim: Dimension of the embedding output
            training_data: Training data to train the model.
            window: The window size to be used to train the model.
            min_count: The minimum number token to be found in training dataset and to be included in the embedding model.
            pretrain_model: The pre-train model to be used.
            trainable: Whether the returned Word embedding model is trainable

        Returns:
            Any: Embedding model.

        """

        if embedding == 'BERT':
            tfhub_handle_encoder = TFTextProcessingConstant.MAP_NAME_TO_HANDLE[pretrain_model]
            tfhub_handle_preprocess = TFTextProcessingConstant.MAP_MODEL_TO_PROCESSES[pretrain_model]

            bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)
            bert_model = hub.KerasLayer(tfhub_handle_encoder, trainable=trainable)

            return bert_model, bert_preprocess_model

        return ImageTextUtil.prepare_embedding_model(
            embedding,
            embedding_dim,
            training_data,
            window,
            min_count,
            pretrain_model,
            trainable
        )

    @staticmethod
    def gensim_to_keras_embedding(model: Any,
                                  train_embeddings: bool = False,
                                  input_shape: Tuple = None):

        """Get a Keras 'Embedding' layer with weights set from Word2Vec model's learned word embeddings.

        Parameters
        ----------
        model: Any
            Gensim word embedding model. This can be Word2vec, FastText or Glove
        train_embeddings : bool
            If False, the returned weights are frozen and stopped from being updated.
            If True, the weights can / will be further updated in Keras.
        input_shape: Tuple
            Input shape to the output layer

        Returns
        -------
        `keras.layers.Embedding`
            Embedding layer, to be used as input to deeper network layers.

        """

        # this is copied from gensim github with additional input shape input as param.
        keyed_vectors = model.wv  # structure holding the result of training
        weights = keyed_vectors.vectors  # vectors themselves, a 2D numpy array

        layer = tf.keras.layers.Embedding(
            input_dim=weights.shape[0],
            output_dim=weights.shape[1],
            weights=[weights],
            trainable=train_embeddings,
            input_shape=input_shape
        )

        return layer

    @staticmethod
    def prepare_image_base_model(model_name: str, image_shape: Tuple):
        """
        Load an image based model (in tensorflow model format) for transfer learning. This will freeze all the layers
        by default, please call set_base_model_trainable if necessary.

        Args:
            model_name: Based model name
            image_shape: Input shape of the model input.

        Returns:
            model: The model in tf.keras.Model or some others acceptable tensorflow model format.

        """

        if model_name == "RestNet50":
            model = tf.keras.applications.ResNet50V2(include_top=False,
                                                     input_shape=image_shape)

        elif model_name == "EfficientNetB0":
            model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(
                input_shape=image_shape,
                include_top=False,
                weights='imagenet')

        elif model_name == "EfficientNetB3":
            model = tf.keras.applications.efficientnet_v2.EfficientNetV2B3(
                input_shape=image_shape,
                include_top=False,
                weights='imagenet')

        model.trainable = False

        return model

    @staticmethod
    def set_base_model_trainable(model: Any, num_trainable_layer: int):
        """
        Unfreeze the model layers except BatchNormalization. For more detail, please check the session of
        Freezing layers: understanding the trainable attribute in
        https://keras.io/guides/transfer_learning/
        Args:
            model: Any tensorflow model.
            num_trainable_layer: Number of layers to be unfreezed. For example, if num_trainable_layer = 50,
                                 the last 50 layers will be set to trainable. To unfreeze all layers, pass in
                                 -1.

        """
        model.trainable = True

        if hasattr(model, "layers"):
            if num_trainable_layer == -1:
                num_trainable_layer = len(model.layers)

            for idx, layer in enumerate(model.layers):
                if idx > (len(model.layers) - num_trainable_layer) and \
                        not isinstance(layer, tf.keras.layers.BatchNormalization) and \
                        not isinstance(layer, keras.layers.normalization.batch_normalization.BatchNormalization):
                    layer.trainable = True
                else:
                    layer.trainable = False
