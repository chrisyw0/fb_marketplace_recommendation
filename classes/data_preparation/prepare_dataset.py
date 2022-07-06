import pandas as pd
import random
from typing import Tuple


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

    def generate_image_product_dataset(self) -> pd.DataFrame:
        """
        Generate an image and product set with joined features
        Returns:
            pd.DataFrame: The joined dataset in panda dataframe format

        """
        df_image_product = pd.merge(self.df_image,
                                    self.df_product,
                                    how="inner",
                                    left_on="product_id",
                                    right_on="id")

        return df_image_product

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

        return dataset.iloc[train_idx], dataset.iloc[val_idx], dataset.iloc[test_idx]
