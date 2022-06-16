import PIL
import numpy as np
import pandas as pd
import pickle

from typing import Tuple, Optional
from PIL import ImageOps


class ImageCleaner:
    """Clean image data
    
    Args:
        df_image (pd.DataFrame): Image dataframe
        df_product (pd.DataFrame): Product dataframe
        cached_path (str, optional): Path to cache the image dataframe. Defaults to "./data/".
        
    """

    def __init__(self,
                 df_image: pd.DataFrame,
                 df_product: pd.DataFrame,
                 cached_path: str = "./data/"
                 ):

        self.df_image = df_image
        self.df_product = df_product
        self.cached_path = cached_path
    
    def clean_image_data(self) -> pd.DataFrame:
        """
        Clean image dataframe by removing images isn't in product frame and width-to-height ratio is < 0.5 or > 1.5 

        Returns:
            pd.DataFrame: Image dataframe
        """
        
        df_image_clean = self.df_image[self.df_image["product_id"].isin(self.df_product["id"].tolist())]
        df_image_clean = df_image_clean[(df_image_clean["image_ratio"] >= 0.5) & (df_image_clean["image_ratio"] <= 1.5)]
        
        return df_image_clean

    def create_image_data(self, path: str, size: Tuple[int, int] = (144, 144)) -> pd.DataFrame:
        """
        Create extra features for image dataframe. The extra/modified features includes: 
        - image_data: Image data in numpy array, the shape is (width, height, 3) 
        - image_width: Width of the orginal image
        - image_height: height of the orginal image
        - image_ratio: Width to height ratio
        - image_mode: The mode of the image "RGB", "RGBA", "L" or "P"

        Args:
            path (str): Folder path storing the images
            size (Tuple[int, int]): The image size to be transformed into numpy array

        Returns:
            pd.DataFrame: _description_
        """
        
        def get_image_information(image_id: str) -> Optional[Tuple[np.ndarray, int, int, float, str]]:
            """
            Retrieve image meta data from the given image ID.

            Args:
                image_id (str): Image ID of the image

            Returns:
                Optional[Tuple[np.ndarray, int, int, float, str]]: Image data, image width, image height, 
                width to height ratio and image mode or None if image is not existed
                
            """
            try:
                image = PIL.Image.open(f"{path}{image_id}.jpg")

                image_size = image.size  
                image_mode = image.mode

                if image.mode != "RGB":
                    image = image.convert("RGB")

                image.thumbnail(size, PIL.Image.LANCZOS)
                image = ImageOps.pad(image, size=size, color=0, centering=(0.5, 0.5))

                return np.asarray(image), image_size[0], image_size[1], image_size[0]/image_size[1], image_mode

            except (FileNotFoundError, PIL.UnidentifiedImageError):
                ...
            
            print(f"Image not found or invalid image")
            return None

        df_image_clean = self.df_image.copy()
        
        print("Converting images")
        df_image_clean["image_pixel_data"], \
            df_image_clean["image_width"],\
            df_image_clean["image_height"], \
            df_image_clean["image_ratio"], \
            df_image_clean["image_mode"] = list(zip(*df_image_clean["id"].apply(get_image_information)))

        print(f"Create images data success, new image dataframe shape {df_image_clean.shape}")
        
        return df_image_clean

    def get_clean_image_data(self) -> pd.DataFrame:
        """
        Get cached image data or clean image data from original data frame

        Returns:
            pd.DataFrame: The image dataframe with removed unused records and additional features
            
        """
        clean_image_path = self.cached_path + "image_clean.pkl"
        
        try:
            with open(clean_image_path, "rb") as f:
                df_image_clean = pickle.load(f)

            print(f"Reload from {clean_image_path} for clean image dataframe")

        except FileNotFoundError:
            self.df_image = self.create_image_data(self.cached_path + "images/")
            df_image_clean = self.clean_image_data()
            
            with open(clean_image_path, "wb") as f:
                pickle.dump(df_image_clean, f)
                
        print(f"Clean data success, new shape {df_image_clean.shape}") 
        print(df_image_clean.head())

        return df_image_clean
