from distutils.command.clean import clean
from typing import Tuple, Optional, List
import pandas as pd
import pickle
import unicodedata
import re
import numpy as np
import PIL
from geopy.geocoders import Nominatim
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from dataclasses import dataclass


@dataclass
class TabularDataCleaner:
    """Clean tabular data
    
    Args:
        df_product (pd.DataFrame): Product dataframe
        df_image (pd.DataFrame): Image dataframe
        cached_path (str, optional): Path of the cached data. Defaults to "./data/".

    """
    df_product: pd.DataFrame
    df_image: pd.DataFrame
    cached_path: str = "./data/"

    def create_product_data(self) -> pd.DataFrame:
        """ Generate additional from original product dataframe and
        merge image data from image dataframe. The extra/modified features includes: 
        
        - lat: Latitude of the location
        - lon: Longitude of the location
        - product_name_description: Combination of product name and product description
        - root_category: The highest level category (root category)
        - sub_category: The lowest level category (i.e. root cat. > sub cat. 1 > sub cat. 2 > ... > (this category))
        - currency: The currency symbol retrieved from the price column
        - price: Price in float data type without currency symbol and formatting characters
        - product_name_tokens: Tokenise the product name column and store them in a token array
        - product_name_word_count: Number of tokens in product name
        - product_description_tokens: Tokenise the product description column and store them in a token array
        - product_description_word_count: Number of tokens in product description
        - product_name_description_tokens: Tokenise the product name and description column and store them in a token array
        - product_name_description_word_count: Number of tokens in product name and description
        - image_data: Merge image data (in numpy array) from image data frame, 
        - image_num: Number of images for the product

        Returns:
            pd.DataFrame: Product dataframe with new features generated or modified features
        """
        
        # initialize components
        geo_app = Nominatim(user_agent="tutorial")

        nlp = English()
        tokenizer = Tokenizer(nlp.vocab)
        
        df_product_clean = self.df_product.copy()
        
        # create a temporary column for later use
        df_product_clean["product_name_temp"] = df_product_clean['product_name'].apply(lambda x : "" if x == "N/A" else x)
        df_product_clean["product_description_temp"] = df_product_clean['product_description'].apply(lambda x : "" if x == "N/A" else x)
        df_product_clean["category_temp"] = df_product_clean['category'].apply(lambda x : "Unknown" if x == "N/A" else x)

        # helper function to parse the currency symbol and price value
        def get_currency_price(prize: str) -> Tuple[str, float]:
            """Parse currency symbol and price from price column 

            Args:
                prize (str): Price in string format

            Returns:
                Tuple[str, float]: Currency symbol (N/A if symbol not found) 
                and price in float format (np.nan if price not found)
                
            """
            symbol = ""
            
            for char in prize:
                if unicodedata.category(char) == "Sc":
                    symbol = char
                    break

            prize_string = re.sub(r"[^\d\.]+", "", prize)
            symbol = symbol if len(symbol) > 0 else "N/A"
            
            try:
                return symbol, float(prize_string)
            except ValueError:
                return symbol, np.nan

        coordinates_map = {}

        location_uk = geo_app.geocode("United Kingdom").raw

        # helper function to get the cooridinate of the location
        def get_coordinates(place: str) -> Tuple[float, float]:
            """Get cooridinate from the location, we assume all the input locations are in the UK

            Args:
                place (str): The location

            Returns:
                Tuple[float, float]: The coordinate (latitude and longitude) of the location
                
            """
            result = coordinates_map.get(place, None)
            if result:
                return result

            result = (location_uk['lat'], location_uk['lon'])

            if place != "N/A":
                try: 
                    location = geo_app.geocode(place + ", United Kingdom").raw
                    result = (location['lat'], location['lon'])
                except:
                    try:
                        county = place.split(",")[-1].strip()
                        location = geo_app.geocode(county + ", United Kingdom").raw
                
                        if "lat" in location and 'lon' in location:
                            result = (location['lat'], location['lon'])
                    except:
                        ...

            coordinates_map[place] = result
            # print(place, result)

            return result 
            
        def get_tokens(text: str) -> Tuple[List[str], int]:
            """Get tokens from a text

            Args:
                text (str): Input text

            Returns:
                Tuple[list, int]: A list of tokens and number of tokens found in the text
            """
            
            tokens = tokenizer(text)
            return [token.text for token in list(tokens)], len(tokens)

        
        def get_image_data(product_id: str) -> Optional[Tuple[List[np.ndarray], int]]:
            """
            Merge image data from image data frame

            Args:
                product_id (str): Product ID

            Returns:
                Optional[Tuple[List[np.ndarray], int]]: None if image data is not existed in image dataframe or 
                list of image (in Numpy array) and number of image 
            """
            df_this_image = self.df_image[self.df_image["product_id"] == product_id]

            try:
                result = []

                for _, image_data in df_this_image["image_pixel_data"].iteritems():
                    result.append(image_data)

                return result, len(result)

            except (IndexError, FileNotFoundError, PIL.UnidentifiedImageError):
                ...
            
            print(f"Image not found or invalid image for product {product_id}")
            return None

        print("Adding extra for product")
        
        # create 3 more features: name and description, root category and sub category
        df_product_clean = df_product_clean.assign(
            product_name_description=(df_product_clean['product_name_temp'] + " " + df_product_clean['product_description_temp']).str.strip(), 
            root_category=df_product_clean['category_temp'].str.split("/").str[0].str.strip(), 
            sub_category=df_product_clean['category_temp'].str.split("/").str[-1].str.strip()
        )

        # tokenizer the product name, description and the combined columns
        df_product_clean["product_name_tokens"], df_product_clean["product_name_word_count"] = list(zip(*df_product_clean["product_name_temp"].apply(get_tokens)))
        df_product_clean["product_description_tokens"], df_product_clean["product_description_word_count"] = list(zip(*df_product_clean["product_description_temp"].apply(get_tokens)))
        df_product_clean["product_name_description_tokens"], df_product_clean["product_name_description_word_count"] = list(zip(*df_product_clean["product_name_description"].apply(get_tokens)))

        print("Mapping images to product")
        df_product_clean["image_data"], df_product_clean["image_num"] = list(zip(*df_product_clean["id"].apply(get_image_data)))

        # remove temporary columns
        df_product_clean.drop(["product_name_temp", "product_description_temp", "category_temp"], axis=1, inplace=True)

        # parsing the price and currency
        df_product_clean["currency"], df_product_clean["price"] = list(zip(*df_product_clean["price"].apply(get_currency_price)))

        print("Adding coordinate to location of the product")
        # getting coordinate for the location
        df_product_clean["lat"], df_product_clean["lon"] = list(zip(*df_product_clean["location"].apply(get_coordinates)))
        
        return df_product_clean

    def get_clean_product_data(self) -> pd.DataFrame:
        """
        Restored image data from cache or clean product data

        Returns:
            pd.DataFrame: Product dataframe 
        """
        # reload dataframe from cached or clean the downloaded data
        clean_product_path = self.cached_path + "product_clean.pkl"
        
        try:
            with open(clean_product_path, "rb") as f:
                df_product_clean = pickle.load(f)

            print(f"Reload from {clean_product_path} for clean product dataframe")

        except FileNotFoundError:
            print("Start cleaning the orginal dataframe")
            df_product_clean = self.create_product_data()

            # remove records which has no name and description
            df_product_clean = df_product_clean[~((df_product_clean['product_name'] == "N/A") & (df_product_clean['product_description'] == "N/A"))]
            
            # remove records with no price or price greater than 1000
            df_product_clean = df_product_clean[df_product_clean['price'] <= 1000]
            
            with open(clean_product_path, "wb") as f:
                pickle.dump(df_product_clean, f)

        print(f"Clean data success, new shape {df_product_clean.shape}") 
        print(df_product_clean.head())
        return df_product_clean
