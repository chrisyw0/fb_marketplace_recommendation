import pandas as pd
import random

from typing import Tuple, List
from sklearn import preprocessing


class DatasetHelper:
    """
    Generate required dataset and manage data split

    Args:
        df_product (pd.DataFrame): Cleaned product dataframe
        df_image (pd.DataFrame): Cleaned product dataframe

        random_state (int, optional): Random state of train and test split. Defaults to 42.
        val_size (float, optional):  Proportion of validation dataset. Defaults to 0.2 (20%).
        test_size (float, optional): Proportion of testing dataset. Defaults to 0.2 (20%).

    """

    def __init__(
            self,
            df_product: pd.DataFrame,
            df_image: pd.DataFrame,
            random_state: int = 42,
            val_size: float = 0.2,
            test_size: float = 0.2
        ):

        self.df_product = df_product
        self.df_image = df_image
        self.random_state = random_state
        self.val_size = val_size
        self.test_size = test_size

    def generate_image_product_dataset(self) -> Tuple[pd.DataFrame, List[str]]:
        """
        Generate an image and product set with joined features, this will also encode the product category
        Returns:
            pd.DataFrame: The joined dataset in panda dataframe format
            List[str]: The unique class name

        """
        df_image_product = pd.merge(self.df_image,
                                    self.df_product,
                                    how="inner",
                                    left_on="product_id",
                                    right_on="id")

        le = preprocessing.LabelEncoder().fit(df_image_product["root_category"].unique())
        category = le.transform(df_image_product["root_category"].tolist())

        df_image_product['category'] = category

        return df_image_product, le.classes_

    def generate_product_data(self) -> pd.DataFrame:
        """
        Generate a product data for price prediction.

        Returns:
            pd.DataFrame: Dataset with required features for price prediction

        """
        # One hot encoding of the root category
        categories = pd.get_dummies(self.df_product["root_category"], prefix="category", drop_first=True)
        df_product_reg = self.df_product.join(categories)

        # drop unused columns
        df_product_reg = df_product_reg.drop(labels=[
            "id", "product_name", "category", "product_description", "product_name_description", \
            "location", "url", "page_id", "create_time", \
            "currency", "product_name_tokens", "product_description_tokens", "product_name_description_tokens", \
            "product_name_description_word_count", "image_path", "root_category", "sub_category"],
            axis=1, errors='ignore'
        )

        return df_product_reg

    def split_dataset(self, dataset: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into training, validation and testing dataset.

        Args:
            dataset (pd.DataFrame): The dataset to be split

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Training, validation and testing dataset

        """

        # set a random_state number to make sure we can reproduce our result
        random.seed(self.random_state)

        # assign index for training, validation and testing dataset
        full_idx = list(range(dataset.shape[0]))
        random.shuffle(full_idx)

        train_end_idx = (int)(dataset.shape[0] * (1 - self.val_size - self.test_size))

        train_idx = full_idx[:train_end_idx]

        val_end_idx = train_end_idx + 1 + (int)(dataset.shape[0] * self.val_size)
        val_idx = full_idx[train_end_idx:val_end_idx]

        test_idx = full_idx[val_end_idx:]

        df_train, df_val, df_test = dataset.iloc[train_idx], dataset.iloc[val_idx], dataset.iloc[test_idx]

        # To deal with multiple images associated with one product once the splitting process is completed
        # it may happen one of this is in training set, the others will also being put into
        # the validation or testing set. This is to avoid the data leaking problem of
        # the same product description is trained and tested in the text understanding model.
        df_train = pd.concat([df_train, df_val[df_val["product_id"].isin(df_train['product_id'].to_list())]])
        df_train = pd.concat([df_train, df_test[df_test["product_id"].isin(df_train['product_id'].to_list())]])

        df_val = df_val[~df_val["product_id"].isin(df_train['product_id'].to_list())]
        df_test = df_test[~df_test["product_id"].isin(df_train['product_id'].to_list())]

        return df_train, df_val, df_test

    @staticmethod
    def get_product_categories(df_train, df_val, df_test) -> Tuple[List[int], List[int], List[int]]:
        y_train = df_train['category'].to_list()
        y_val = df_val['category'].to_list()
        y_test = df_test['category'].to_list()

        return y_train, y_val, y_test

    @staticmethod
    def get_image_ids(df_train, df_val, df_test) -> Tuple[List[str], List[str], List[str]]:
        image_train = df_train['id_x'].to_list()
        image_val = df_val['id_x'].to_list()
        image_test = df_test['id_x'].to_list()

        return image_train, image_val, image_test


