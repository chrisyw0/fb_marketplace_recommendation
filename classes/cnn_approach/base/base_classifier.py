import pandas as pd
from typing import Tuple, List, Optional, Any


class BaseClassifier:
    """
    This is the base class of our deep learning model which provides interface for
    methods and attributes, and standard methods.

    It contains the following majors methods, some of them should be implemented in the subclass:
    1. prepare_data - Input product and image data and get the training, validation and testing dataset.
    2. process_load_model - Set up essential parameters before loading a model, this should be the first step for load
                            model process.
    3. create_model - Create a deep learning model according to the input data shape and other configuration
    4. show_model_summary - Show the summary of the model, as well as export the model graph.
    5. train_model - Train the model with training dataset, and validate it with validation dataset.
    6. fine_tune_model - Fine tune the model with the same dataset, unfreeze some layers and change the learning rate
                         if necessary. (Optional)
    7. evaluate_model/predict_model - Test the model with the testing dataset. evaluate_model will only get the
                                      overall accuracy and loss while predict_model will return a classification
                                      report and predicted labels for the testing dataset.
    8. visualise_performance - Plot the accuracy and loss in each epoch for training and validation dataset
    9. save_model - Save the weight for the model for later use.
    10. clean_up - remove the folders storing the images.

    The method "process" will run through the whole process, should be the entry point if you want to train and
    test the model.

    Attributes

    model_name(str): Name of the model. This parameter can be updated by overriding method _get_model_name
    log_path(str): Path for storing the logs, which will be later uploaded into tensorboard to visualise the model
                   performance. Default: f"./logs/{self._get_model_name()}/"
    model_path(str): Path for storing the model weights. Once the model is trained, the weight will be available
                     to load from ckpt files stored in this path. Default: f"./model/{self._get_model_name()}/weights/"
    learning_rate(float): Learning rate to be used in the training stage.
    epoch(int): Number of epoch to be used in the training stage.
    batch_size(int): The number of records to be inputted to the model at a time.
    metrics(List[str]): The metrics to be used to evaluate the model performance

    """
    model_name: str

    log_path: str
    model_path: str

    learning_rate: float
    epoch: int
    batch_size: int

    metrics: List[str]

    model: Optional[Any] = None
    history: Optional[Any] = None

    def __init__(self, df_image: pd.DataFrame, df_product: pd.DataFrame):
        self.df_image = df_image
        self.df_product = df_product

    def _get_model_name(self) -> str:
        """
        Return the model name
        This method optional to be implemented in child class.

        Returns:
            model name in str format

        """
        return "model"

    def process(self) -> None:
        """
        The entry point of the process. It runs through all required steps of training and testing stage.
        """

        self.model_name = self._get_model_name()

        self.log_path: str = f"./logs/{self._get_model_name()}/"
        self.model_path: str = f"./model/{self._get_model_name()}/weights/"

        self.prepare_data()
        self.create_model()
        self.show_model_summary()
        self.train_model()
        self.fine_tune_model()
        self.evaluate_model()
        self.save_model()
        self.visualise_performance()
        self.clean_up()

    def process_load_model(self):
        """
        This is the setup method before loading a model.

        """
        self.model_name = self._get_model_name()

        self.log_path: str = f"./logs/{self._get_model_name()}/"
        self.model_path: str = f"./model/{self._get_model_name()}/weights/"

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create features, normalise data, and split data into training, validation and testing dataset.
        This method should be implemented in child class.
        """
        pass

    def create_model(self):
        """
        Create the CNN Model.
        This method should be implemented in child class.
        """
        pass

    def show_model_summary(self):
        """
        Show model summary
        This method should be implemented in child class.
        """
        pass

    def train_model(self):
        """
        Train the Model.
        This method should be implemented in child class.
        """
        pass

    def fine_tune_model(self):
        """
        Fine tune the model if necessary
        This method should be implemented in child class.
        """

        pass

    def evaluate_model(self) -> Tuple[float, float]:
        """
        Evaluate the CNN Model for the testing dataset.
        This method should be implemented in child class.
        """
        pass

    def predict_model(self, data: Any) -> List[int]:
        """
        Predict from the CNN Model.
        This method should be implemented in child class.
        """
        pass

    def load_model(self):
        """
        Create a model with saved weight
        This method should be implemented in child class.
        """
        pass

    def save_model(self):
        """
        Save weight of the trained model.
        This method should be implemented in child class.
        """
        pass

    def visualise_performance(self) -> None:
        """
        Visual the performance of the model. It will plot loss and accuracy for training and validation dataset
        in each epoch.
        This method should be implemented in child class.
        """

        pass

    def clean_up(self) -> None:
        """
        Clear the tensorflow backend session
        This method should be implemented in child class.
        """
        pass
