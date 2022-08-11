import os
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.utils.vis_utils import plot_model
from classes.dl.base.base_classifier import BaseClassifier
from official.nlp import optimization


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


def get_optimizer(
        ds_train: tf.data.Dataset,
        epoch: int,
        learning_rate: float,
        optimizer_type: str = "adamw",
        warmup_ratio: float = 0.1
) -> tf.keras.optimizers.Optimizer:
    """
    Get an optimizer built with a polynomial decay scheduler.
    Args:
        ds_train: Training dataset, use to calculate number of steps in each epoch.
        epoch: Number of epochs to be used in training
        learning_rate: Learning rate of each optimization step
        optimizer_type: The optimizer type, defaults to adamw. See official.nlp.optimization.create_optimizer for
                        possible value
        warmup_ratio: The percentage of data to be used in warmup stage, defaults to 0.1.

    Returns:
        tf.keras.optimizers.Optimizer: An optimizer with learning rate decay scheduler.

    """

    steps_per_epoch = tf.data.experimental.cardinality(ds_train).numpy()
    num_train_steps = steps_per_epoch * epoch
    num_warmup_steps = int(warmup_ratio * num_train_steps)

    optimizer = optimization.create_optimizer(init_lr=learning_rate,
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type=optimizer_type)

    return optimizer
