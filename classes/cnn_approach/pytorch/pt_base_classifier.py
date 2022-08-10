import os
import gc
import inspect
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import transformers

from tqdm import tqdm
from classes.cnn_approach.base.base_classifier import BaseClassifier
from sklearn.metrics import classification_report
from torchinfo import summary
from typing import Tuple, Dict, Union, List, Any
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# this is to cater the availability of gpu device. "mps" is metal M1 device, "cuda" is Nvidia GPU and fallback
# option is to use cpu
pt_device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


class PTBaseClassifier(BaseClassifier):
    device = pt_device
    skip_summary = False
    input_data: Any = None
    input_shape: Union[Tuple, Dict]
    input_dtypes: List
    classes: List[str]
    num_class: int
    train_dl: DataLoader
    val_dl: DataLoader
    test_dl: DataLoader

    """
    This is the PyTorch version of the BaseClassifier with several common methods implemented
    """

    def show_model_summary(self):
        """
        Show model summary
        """

        if not self.skip_summary:
            if self.input_data:
                print(summary(
                    self.model,
                    input_data=self.input_data,
                    device=pt_device
                ))
            else:
                print(summary(
                    self.model,
                    self.input_shape,
                    dtypes=self.input_dtypes,
                    batch_dim=0,
                    device=pt_device
                ))

    def visualise_performance(self):
        """
        Visual the performance of the model. It will plot loss and accuracy for training and validation dataset
        in each epoch.

        """

        if self.history is None:
            raise ValueError("[Error] Training history not found, please train your model before calling this function")

        # plot the loss
        plt.plot(self.history['loss'], label='train loss')
        plt.plot(self.history['val_loss'], label='val loss')
        plt.legend()
        plt.show()

        # plot the accuracy
        plt.plot(self.history['accuracy'], label='train acc')
        plt.plot(self.history['val_accuracy'], label='val acc')
        plt.legend()
        plt.show()

    def load_model(self):
        """
        Create a model with saved weight
        """

        self.process_load_model()
        self.prepare_data()
        self.create_model()
        self.model.load_state_dict(torch.load(f"{self.model_path}model.pt"))

    def save_model(self):
        """
        Save weight of the trained model.
        """

        if self.model is None:
            raise ValueError("[Error] Model not found, please create your model before calling this function")

        os.makedirs(self.model_path, exist_ok=True)
        torch.save(self.model.state_dict(), f"{self.model_path}model.pt")

    def clean_up(self) -> None:
        """
        Clear the memory
        """

        torch.cuda.empty_cache()
        gc.collect()


def _process_input_data(
        input_data: List[torch.Tensor],
        args_list: List[str],
        to_device_input_data: bool = True,
        to_device_label: bool = False
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Process the input data, converting to a dictionary, and convert the input tensor to match the device. The last
    element will always map to label in the returned value. This function is intended to map the input data to a
    torch model forward method as different model will have different input arguments. (i.e. image only, text only,
    image and text)

    e.g. input_data = [tensor1, tensor2, tensor3], args_list = ["image", "text", "label"]
    > ({"image": tensor1, "text": tensor2},  tensor3)

    Args:
        input_data: A list of input data in the type of torch.Tensor
        args_list: A list of arguments retrieved from the model forward method.
        to_device_input_data: Whether to convert the input_data tensor according the GPU availability
        to_device_label: Whether to convert the label tensor according the GPU availability

    Returns:
        Dict[str, torch.Tensor]: A dictionary contains keys "text" and/or "image" and input data in
                                 Pytorch tensor format. If to_device_input_data = True,
                                 the tensor in each value is set to match the device
                                 according the GPU availability.
        torch.Tensor: Label in Pytorch tensor format. If to_device_label = True,
                      the tensor in each value is set to match the device according the GPU availability.

    """
    x0 = input_data[0]

    if len(input_data) > 2:
        x1 = input_data[1]
        inputs = {
            "image": x0,
            "text": x1
        }

        labels = input_data[2]

    else:
        if "image" in args_list:
            inputs = {"image": x0}
        elif "text" in args_list:
            inputs = {"text": x0}

        labels = input_data[1]

    if to_device_input_data:
        for key, value in inputs.items():
            if isinstance(value, dict):
                for sub_key, sub_value in inputs[key].items():
                    inputs[key][sub_key] = sub_value.to(pt_device)
            else:
                inputs[key] = value.to(pt_device)

    if to_device_label:
        labels = labels.to(pt_device)

    return inputs, labels


def prepare_optimizer_and_scheduler(
        model: nn.Module,
        total_samples: int,
        batch_size: int,
        learning_rate: float,
        epoch: int,
        warm_up_ratio: float = 0.1
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    """
    Get an AdamW optimizer and a polynomial decay scheduler.
    Args:
        model: The Pytorch model
        total_samples: Total number of records of your dataset.
        batch_size: Number of records to be loaded in a single batch.
        learning_rate: Learning rate of each optimization step.
        epoch: Number of epoch to be used for training.
        warm_up_ratio: The percentage of data to be used in warmup stage, defaults to 0.1.

    Returns:
        torch.optim.Optimizer: AdamW optimizer setup with model parameters and learning rate
        torch.optim.lr_scheduler.LambdaLR: The polynomial learning rate decay scheduler.

    """
    optimizer = transformers.optimization.AdamW(
        model.parameters(),
        lr=learning_rate
    )

    steps_per_epoch = total_samples // batch_size

    num_warmup_steps = int(steps_per_epoch * warm_up_ratio)
    num_training_steps = steps_per_epoch * epoch

    scheduler = transformers.get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps)

    return optimizer, scheduler


def train_and_validate_model(
        model: nn.Module,
        train_dl: DataLoader,
        val_dl: DataLoader,
        criterion: Union[nn.CrossEntropyLoss, nn.BCELoss],
        epoch: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR = None,
        summary_writer: SummaryWriter = None,
        init_epoch: int = 1
) -> Dict[str, List[float]]:
    """
    Perform model training and validation. It processes the input data from training and validation dataloader, trains
    and validate the model, reports the accuracy and loss in each epoch.
    Args:
        model: The Pytorch model
        train_dl: Dataloader of the training dataset
        val_dl: Dataloader of the validation dataset
        criterion: Loss function to be used. Can be either nn.CrossEntropyLoss for multi-class classification or
                   nn.BCELoss for binary class classification
        epoch: Number of epoch to be used in model training stage
        optimizer: Optimizer to be used in model training stage
        scheduler: Scheduler to be used in model training stage, None means we don't use scheduler to decrease
                   learning rate in the model training stage.
        summary_writer: Tensorboard summary writer to write training logs that can be sent to Tensorboard
        init_epoch: The first epoch to be written into tensorboard log. It is usually 1 when the model hasn't been
                    trained before. It is useful when we fine-tune the model where the first epoch will be equal
                    to the number of epoch being trained.

    Returns:
        Dict[str, List[float]]: A dictionary contains 4 keys and values:
                            "loss": Training loss in each epoch
                            "val_loss": Validation loss in each epoch
                            "accuracy": Training accuracy in each epoch
                            "val_accuracy": Validation accuracy in each epoch

    """

    result_train_loss = []
    result_train_accuracy = []
    result_val_loss = []
    result_val_accuracy = []

    for epoch in range(epoch):
        # start training
        model.train()

        pred_labels = []
        actual_labels = []

        running_train_loss = 0.0
        count = 0

        # read the input args of the model forward method, we will use the list to match the input data
        args_list = inspect.getfullargspec(model.forward).args

        for i, data in enumerate(tqdm(train_dl)):  # tqdm gives you a nice progress bar
            # for every batch in training dataset
            # zero the parameter gradients
            optimizer.zero_grad()

            # the input is a dictionary with keys matching to the model forward method,
            # we can use **input to pass the input data from a dictionary
            inputs, labels = _process_input_data(data, args_list)
            outputs = model(**inputs)

            # the model doesn't have softmax activation in the final layer, we need this to find out the prediction
            # so that we can check the accuracy
            this_pred = F.softmax(outputs, dim=1)
            this_pred = this_pred.cpu().detach().numpy()
            this_pred = [np.argmax(x) for x in this_pred]

            pred_labels.extend(this_pred)
            y_true = labels.cpu().detach().tolist()
            actual_labels.extend(y_true)

            labels = labels.to(pt_device)

            # calculate the loss
            loss = criterion(outputs, labels)

            # backpropagation, adjust the weight according to the loss changes
            loss.backward()

            # a step forward for optimizer and scheduler, this may adjust learning rate specifically
            # applied to different parameters in the model.
            optimizer.step()
            if scheduler:
                scheduler.step()

            running_train_loss += loss.item()
            count += 1

        # calculate the avg loss and accuracy after training all batches of data
        train_loss = running_train_loss / count
        train_acc = np.mean(np.array([pred_labels[i] == actual_labels[i] for i in range(len(actual_labels))]))

        # start evaluate the model
        model.eval()

        running_val_loss = 0.0
        count = 0

        val_pred_labels = []
        val_actual_labels = []

        for i, data in enumerate(val_dl, 0):
            # for every batch in validation dataset
            inputs, labels = _process_input_data(data, args_list)

            # we don't do any gradient descent for validation dataset. The stage will input the data into the model and
            # calculate the loss and accuracy of the dataset.
            with torch.no_grad():
                outputs = model(**inputs)

                this_pred = F.softmax(outputs, dim=1)
                this_pred = this_pred.cpu().detach().numpy()

                val_pred_labels.extend([np.argmax(x) for x in this_pred])

                actual_labels = labels.cpu().detach().tolist()
                val_actual_labels.extend(actual_labels)

                labels = labels.to(pt_device)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item()

            count += 1

        # calculate avg. loss and accuracy for validation dataset.
        val_loss = running_val_loss / count
        val_acc = np.mean(
            np.array([val_pred_labels[i] == val_actual_labels[i] for i in range(len(val_actual_labels))]))

        result_train_loss.append(train_loss)
        result_val_loss.append(val_loss)
        result_train_accuracy.append(train_acc)
        result_val_accuracy.append(val_acc)

        current_epoch = epoch + init_epoch

        # write the loss and accuracy to the log
        if summary_writer:
            summary_writer.add_scalar('Loss/train', train_loss, current_epoch)
            summary_writer.add_scalar('Loss/validation', val_loss, current_epoch)
            summary_writer.add_scalar('Accuracy/train', train_acc, current_epoch)
            summary_writer.add_scalar('Accuracy/validation', val_acc, current_epoch)

        print('Epoch {} loss: {:.2f}, accuracy: {:.2f} val loss: {:.2f}, val accuracy {:.2f}'.format(
            current_epoch,
            train_loss,
            train_acc,
            val_loss,
            val_acc
        ))

        # end of epoch

    return {
        "loss": result_train_loss,
        "val_loss": result_val_loss,
        "accuracy": result_train_accuracy,
        "val_accuracy": result_val_accuracy
    }


def evaluate_model(
        model: nn.Module,
        test_dl: DataLoader,
        criterion: Union[nn.CrossEntropyLoss, nn.BCELoss],
        class_name: List[str]
) -> Tuple[float, float]:
    """
    Evaluate the model with a testing dataset. This function is designed for dataset that contains labels. It will
    get the prediction from the model, and compare with the true labels, and finally giving the accuracy, loss and
    print the classification report to the dataset.

    Args:
        model: The model to be evaluated
        test_dl: The dataloader of testing dataset
        criterion: Loss function to be used. Can be either nn.CrossEntropyLoss for multi-class classification or
                   nn.BCELoss for binary class classification
        class_name: A list of class name, will be useful for printing the classification report.

    Returns:
        float: Loss of testing dataset given by the model.
        float: Accuracy of testing dataset given by the model.

    """
    model.eval()

    running_test_loss = 0.0
    count = 0

    test_pred_labels = []
    test_actual_labels = []

    args_list = inspect.getfullargspec(model.forward).args

    for i, data in enumerate(test_dl, 0):
        inputs, labels = _process_input_data(data, args_list)

        with torch.no_grad():
            outputs = model(**inputs)
            this_pred = outputs.cpu().detach().numpy()

            test_pred_labels.extend([np.argmax(x) for x in this_pred])
            actual_labels = labels.cpu().detach().tolist()
            test_actual_labels.extend(actual_labels)

            labels = labels.to(pt_device)

            loss = criterion(outputs, labels)
            running_test_loss += loss.item()

        count += 1

    test_loss = running_test_loss / count
    test_acc = np.mean(np.array([test_pred_labels[i] == test_actual_labels[i] for i in range(len(test_actual_labels))]))

    report = classification_report(test_actual_labels, test_pred_labels, target_names=class_name)
    print(report)

    print("Evaluation on model accuracy {:.2f}, loss {:.2f}".format(test_acc, test_loss))
    return test_loss, test_acc


def predict_model(
        model: nn.Module,
        dataloader: DataLoader,
) -> List[int]:
    """
    Get prediction for the dataset. This function is designed for dataset which doesn't contain any true label. It
    simply gives the predicted labels from the model.
    Args:
        model:
        dataloader:

    Returns:
        List[int]: prediction labels

    """
    model.eval()
    y_pred = []

    args_list = inspect.getfullargspec(model.forward).args

    for i, data in enumerate(dataloader, 0):
        args = data
        x0 = args[0].to(pt_device)

        if len(args) > 2:
            x1 = args[1].to(pt_device)
            inputs = {
                "image": x0,
                "text": x1
            }
        elif "image" in args_list:
            inputs = {"image": x0}
        elif "text" in args_list:
            inputs = {"text": x0}

        outputs = model(**inputs)
        outputs = F.softmax(outputs, dim=1)

        this_pred = outputs.cpu().detach().numpy()

        y_pred.extend(np.argmax(x) for x in this_pred)

    return y_pred
