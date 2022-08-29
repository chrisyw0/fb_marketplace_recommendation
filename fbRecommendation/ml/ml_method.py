import pandas as pd
import numpy as np
import PIL

from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, classification_report
from typing import Tuple

from fbRecommendation.dataset.prepare_dataset import DatasetHelper


class MachineLearningPredictor:
    """Predict price and category using machine learning models
    
    Args:
        df_product (pd.DataFrame): Product dataframe with additional features generated
        df_image (pd.DataFrame): Image dataframe with additional features generated
    """

    def __init__(self,
                 df_product: pd.DataFrame,
                 df_image: pd.DataFrame
                 ):

        self.df_product = df_product
        self.df_image = df_image

    def predict_price(self) -> Tuple[LinearRegression, float]:
        """
        Train and predict price from product information by using a linear regression model. 
        The whole process includes generating features (one hot vector), normalising data,
        spliting the dataset into training and testing dataset, train a model with training dataset 
        and predict the price for testing dataset with the model. It returns the model and RMSE

        Returns:
            Tuple[LinearRegression, float]: Linear regression model and RMSE
        """

        # get the dataset
        data_generator = DatasetHelper(self.df_product, self.df_image)
        df_product = data_generator.generate_product_data()

        # we don't use validation dataset for ml model
        df_train, _, df_test = data_generator.split_dataset(df_product)

        y_train = df_train.price.tolist()
        y_test = df_test.price.tolist()

        X_train = df_train.drop("price", axis=1).to_numpy()
        X_test = df_test.drop("price", axis=1).to_numpy()

        # use scaler to normalise the features
        scaler = preprocessing.MinMaxScaler()
        
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # train a linear regression with training dataset
        reg = LinearRegression().fit(X_train, y_train)
        
        # predict the result with the trained model
        y_pred = reg.predict(X_test)
        
        # use RMSE to measure the performance of the model
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        
        print("Linear Regression Result:")
        print("="*40)
        print(f"Coef: {reg.coef_}")
        print(f"Interception: {reg.intercept_}")
        print(f"RMSE: {rmse}")
        print("="*40)
        
        return reg, rmse

    def predict_product_type(self) -> Tuple[LogisticRegression, dict]:
        """
        Train and predict the product type from image information using a logistic regression model. 
        The whole process includes generating features (one hot vector), normalising data,
        splitting the dataset into training and testing dataset, train a model with training dataset
        and predict the price for testing dataset with the model. It returns the model and classification report

        Returns:
            Tuple[LogisticRegression, float]: Logistic regression model and classification report in dictionary format
        """

        data_generator = DatasetHelper(self.df_product, self.df_image)
        df_image_product = data_generator.generate_image_product_dataset()

        # print(df_image_product.columns)

        def _convert_image(path):
            image = PIL.Image.open(path)
            return np.asarray(image)

        # apply flattening to image data, making the data as features
        image_data = df_image_product["adjust_image_file"].apply(lambda x: _convert_image(x).flatten()).tolist()
        image_data = np.asarray(image_data)
        
        df_image_pixel = pd.DataFrame(image_data, columns=[f"pixel_{i}" for i in range(image_data.shape[1])])
        
        # use one hot encoding to encode the image mode
        image_mode = pd.get_dummies(df_image_product["image_mode"], prefix="image_mode", drop_first=True)
        
        df_image_product = df_image_product.join(image_mode)

        # use label encoder to encode the type of the product
        le = preprocessing.LabelEncoder().fit(df_image_product["root_category"].unique())
        category = le.transform(df_image_product["root_category"].tolist())

        df_image_product['category'] = category
        
        # include features with numeric value only 
        df_image_log = df_image_product[["image_width", "image_height", "image_mode_P",
                                         "image_mode_RGB", "image_mode_RGBA", 'category']]
        df_image_log = df_image_log.join(df_image_pixel)
        
        # split the training and testing dataset
        df_train, _, df_test = data_generator.split_dataset(df_image_log)

        X_train = df_train.drop("category", axis=1).to_numpy()
        X_test = df_test.drop("category", axis=1).to_numpy()

        y_train = df_train.category.tolist()
        y_test = df_test.category.tolist()

        # use scaler to normalise the features
        scaler = preprocessing.MinMaxScaler()
        
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # train a logistic regression with training dataset
        log = LogisticRegression()
        log.fit(X_train, y_train)
        
        # predict the result with the trained model
        y_pred = log.predict(X_test)
        
        # generate classification report 
        result = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
        
        print("Logistic Regression Result:")
        print("="*40)
        print(f"Coef: {log.coef_}")
        print(f"Interception: {log.intercept_}")
        print(f"{classification_report(y_test, y_pred, target_names=le.classes_)}")
        print("="*40)
        
        return log, result
