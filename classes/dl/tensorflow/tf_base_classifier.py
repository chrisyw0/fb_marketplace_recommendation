import os
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.utils.vis_utils import plot_model
from classes.dl.base.base_classifier import BaseClassifier


class TFBaseClassifier(BaseClassifier):
    """
    This is the tensorflow version of the BaseClassifier with several common methods implemented
    """
    def show_model_summary(self) -> None:
        """
        Show model summary
        """

        print(self.model.summary(expand_nested=True, show_trainable=True))

        os.makedirs('./model/', exist_ok=True)

        plot_model(self.model,
                   to_file=f'./model/{self.model_name}.png',
                   show_shapes=True,
                   show_layer_names=True)

    def visualise_performance(self) -> None:
        """
        Visual the performance of the model. It will plot loss and accuracy for training and validation dataset
        in each epoch.

        """

        if self.history is None:
            raise ValueError("[Error] Training history not found, please train your model before calling this function")

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

    def load_model(self):
        """
        Create a model with saved weight
        """

        self.create_model()
        self.model.load_weights(f"{self.model_path}model.ckpt")

    def save_model(self):
        """
        Save weight of the trained model.
        """

        if self.model is None:
            raise ValueError("[Error] Model not found, please create your model before calling this function")

        self.model.save_weights(f"{self.model_path}model.ckpt")

    def clean_up(self) -> None:
        """
        Clear the tensorflow backend session
        """

        tf.keras.backend.clear_session()
