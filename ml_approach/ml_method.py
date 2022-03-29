import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, classification_report
from typing import Tuple

def predict_price(df_product: pd.DataFrame, random_state: int = 42, test_size: float = 0.3) -> Tuple[LinearRegression, float]:
    """
    Train and predict price from product information by using a linear regression model. 
    The whole process includes generating features (one hot vector), normalising data,
    spliting the dataset into training and testing dataset, train a model with training dataset 
    and predict the price for testing dataset with the model. It returns the model and RMSE

    Args:
        df_product (pd.DataFrame): Product dataframe with additional features generated
        random_state (int, optional): Random state of train and test split. Defaults to 42.
        test_size (float, optional): Proportion of testing dataset. Defaults to 0.3 (30%).
    
    Returns:
        Tuple[LinearRegression, float]: Linear regression model and RMSE
    """
    
    # Retrieve the price column as the prediciton target
    df_price = df_product[["price"]]
    product_price = df_price["price"].tolist()
    
    # One hot encoding of the root category
    categories = pd.get_dummies(df_product["root_category"], prefix="category", drop_first=True)
    df_product_reg = df_product.join(categories)
    
    # drop unused columns
    df_product_reg = df_product_reg.drop(labels=[
        "id", "product_name", "category", "product_description", "product_name_description", \
        "location", "url", "page_id", "create_time", "price", \
        "currency", "product_name_tokens", "product_description_tokens", "product_name_description_tokens", \
        "product_name_description_word_count", "image_data", "root_category", "sub_category"], axis=1)
    
    # split the training and testing dataset
    X_train, X_test, y_train, y_test = train_test_split(df_product_reg, product_price, test_size=test_size, random_state=random_state)
    
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

def predict_product_type(df_product: pd.DataFrame, df_image: pd.DataFrame, random_state: int = 42, test_size: float = 0.3) -> Tuple[LogisticRegression, dict]:
    """
    Train and predict the product type from image information using a logistic regression model. 
    The whole process includes generating features (one hot vector), normalising data,
    spliting the dataset into training and testing dataset, train a model with training dataset 
    and predict the price for testing dataset with the model. It returns the model and classification report

    Args:
        df_product (pd.DataFrame): Product dataframe with additional features generated
        df_image (pd.DataFrame): Image dataframe with additional features generated
        random_state (int, optional): Random state of train and test split. Defaults to 42.
        test_size (float, optional): Proportion of testing dataset. Defaults to 0.3 (30%).
        
    Returns:
        Tuple[LogisticRegression, float]: Logistic regression model and classification report in dictionary format
    """
    
    # use label encoder to encode the type of the product
    le = preprocessing.LabelEncoder().fit(df_product["root_category"].unique())
    category = le.transform(df_product["root_category"].tolist())
    
    df_product["category"] = category
    
    # apply flattening to image data, making the data as features
    image_data = df_image["image_data"].apply(lambda x : x.flatten()).tolist()
    image_data = np.asarray(image_data)
    
    df_image_pixel = pd.DataFrame(image_data, columns=[f"pixel_{i}" for i in range(image_data.shape[1])])
    
    # use one hot encoding to encode the image mode
    image_mode = pd.get_dummies(df_image["image_mode"], prefix="image_mode", drop_first=True)
    
    df_image_data = df_image.join(image_mode)
    df_image_data = pd.merge(df_image_data, df_product, how="inner", left_on="product_id", right_on="id")
    
    # include features with numeric value only 
    df_image_log = df_image_data[["image_width", "image_height", "image_mode_P", "image_mode_RGB", "image_mode_RGBA"]]
    df_image_log = df_image_log.join(df_image_pixel)
    
    # select category as prediction target
    df_category = df_image_data[["category"]]
    image_categories = df_category["category"].tolist()
    
    # split the training and testing dataset
    X_train, X_test, y_train, y_test = train_test_split(df_image_log, image_categories, test_size=test_size, random_state=random_state)
    
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