import os
import gc
import inspect
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from classes.cnn_approach.base.base_classifier import BaseClassifier
from sklearn.metrics import classification_report
from torchsummary import summary


pt_device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


class PTBaseClassifier(BaseClassifier):
    device = pt_device
    """
    This is the tensorflow version of the BaseClassifier with several common methods implemented
    """

    def show_model_summary(self) -> None:
        """
        Show model summary
        """
        summary(
            self.model,
            self.input_shape,
            dtypes=self.input_dtypes,
            batch_dim=0,
            device=pt_device
        )

    def visualise_performance(self) -> None:
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
        Clear the tensorflow backend session
        """

        torch.cuda.empty_cache()
        gc.collect()


def train_and_validate_model(
        model,
        train_dl,
        val_dl,
        criterion,
        epoch,
        optimizer,
        scheduler=None,
        summary_writer=None,
        init_epoch=1
):
    result_train_loss = []
    result_train_accuracy = []
    result_val_loss = []
    result_val_accuracy = []

    for epoch in range(epoch):
        model.train()

        pred_labels = []
        actual_labels = []

        running_train_loss = 0.0
        count = 0

        args_list = inspect.getfullargspec(model.forward).args

        for i, data in enumerate(tqdm(train_dl)):
            # zero the parameter gradients
            optimizer.zero_grad()

            args = data
            x0 = args[0].to(pt_device)

            if len(args) > 2:
                x1 = args[1].to(pt_device)
                inputs = {
                    "image": x0,
                    "text": x1
                }

                labels = args[2]
            else:
                if "image" in args_list:
                    inputs = {"image": x0}
                elif "text" in args_list:
                    inputs = {"text": x0}

                labels = args[1]

            outputs = model(**inputs)

            this_pred = F.softmax(outputs, dim=1)
            this_pred = this_pred.cpu().detach().numpy()
            this_pred = [np.argmax(x) for x in this_pred]

            pred_labels.extend(this_pred)
            y_true = labels.cpu().detach().tolist()
            actual_labels.extend(y_true)

            labels = labels.to(pt_device)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if scheduler:
                scheduler.step()

            running_train_loss += loss.item()
            count += 1

        train_loss = running_train_loss / count
        train_acc = np.mean(np.array([pred_labels[i] == actual_labels[i] for i in range(len(actual_labels))]))

        model.eval()

        running_val_loss = 0.0
        count = 0

        val_pred_labels = []
        val_actual_labels = []

        for i, data in enumerate(val_dl, 0):
            args = data
            x0 = args[0].to(pt_device)

            if len(args) > 2:
                x1 = args[1].to(pt_device)
                inputs = {
                    "image": x0,
                    "text": x1
                }

                labels = args[2]
            else:
                if "image" in args_list:
                    inputs = {"image": x0}
                elif "text" in args_list:
                    inputs = {"text": x0}

                labels = args[1]

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

        val_loss = running_val_loss / count
        val_acc = np.mean(
            np.array([val_pred_labels[i] == val_actual_labels[i] for i in range(len(val_actual_labels))]))

        result_train_loss.append(train_loss)
        result_val_loss.append(val_loss)
        result_train_accuracy.append(train_acc)
        result_val_accuracy.append(val_acc)

        current_epoch = epoch + init_epoch

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

    return {
        "loss": result_train_loss,
        "val_loss": result_val_loss,
        "accuracy": result_train_accuracy,
        "val_accuracy": result_val_accuracy
    }


def evaluate_model(
        model,
        test_dl,
        criterion,
        class_name
):
    model.eval()

    running_test_loss = 0.0
    count = 0

    test_pred_labels = []
    test_actual_labels = []

    args_list = inspect.getfullargspec(model.forward).args

    for i, data in enumerate(test_dl, 0):
        args = data
        x0 = args[0].to(pt_device)

        if len(args) > 2:
            x1 = args[1].to(pt_device)
            inputs = {
                "image": x0,
                "text": x1
            }

            labels = args[2]
        else:
            if "image" in args_list:
                inputs = {"image": x0}
            elif "text" in args_list:
                inputs = {"text": x0}

            labels = args[1]

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
        model,
        dataloader,
):
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
