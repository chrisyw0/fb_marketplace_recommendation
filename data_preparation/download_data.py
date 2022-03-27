import os
import yaml
import re
import pickle
import pandas as pd
import requests
import fnmatch
from sqlalchemy import create_engine
from typing import Tuple
from tqdm import tqdm
from zipfile import ZipFile

def read_cloud_config(config_file : str) -> dict:
    """
    Helper function to read RDS config. For environmental varaibles, 
    use ${ENV_NAME} where ENV_NAME is the name of environmental variable.

    Args: 
        config_file (str): File path for the config file

    Returns:
        dict: Config in dictionary format

    """
    env_pattern = re.compile(r".*?\${(.*?)}.*?")
    def env_constructor(loader, node):
        value = loader.construct_scalar(node)
        for group in env_pattern.findall(value):
            # print(group, os.environ.get(group))
            value = value.replace(f"${{{group}}}", os.environ.get(group))
        return value

    yaml.SafeLoader.add_implicit_resolver("!pathex", env_pattern, None)
    yaml.SafeLoader.add_constructor("!pathex", env_constructor)

    # read config
    f = open(config_file, "r")
    aws_config = yaml.safe_load(f)
    
    return aws_config

def download_data(cached_path: str = "./data/", config_file: str = "./aws.yaml") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Download product and image database from postgres DB and read them into dataframes

    Args:
        cached_path (str, optional): The path of the cached folder. Defaults to "./data/".
        config_file (str, optional): The path of the config file. Defaults to "./aws.yaml".

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Dataframe of product and images
    """
    os.makedirs(cached_path, exist_ok=True)    
    
    product_path = cached_path + "product.pkl"
    image_path = cached_path + "image.pkl"

    try:
        print("Try reloading from Google Drive for product and image dataframe")
        with open(product_path, "rb") as f:
            df_product = pickle.load(f)

        with open(image_path, "rb") as f:
            df_image = pickle.load(f)

    except FileNotFoundError:
        print("DataFrame not found, download from DB")
        
        config = read_cloud_config(config_file)["aws-postgresql"]

        db_type = config["dbtype"]
        db_api = config["dbapi"]
        user = config["username"]
        password = config["password"]
        endpoint = config["endpoint"]
        port = config["port"]
        database = config["database"]

        engine = create_engine(f"{db_type}+{db_api}://{user}:{password}@{endpoint}:{port}/{database}")
        engine.connect()

        df_product = pd.read_sql_table("products", engine)
        df_image = pd.read_sql_table("images", engine)

        with open(product_path, "wb") as f:
            pickle.dump(df_product, f)

        with open(image_path, "wb") as f:
            pickle.dump(df_image, f)
     
    print(f"Load data success, tabular data shape {df_product.shape}, image data shape {df_image.shape}")
    return df_product, df_image

def download_images(cached_path: str = "./data/", config_file: str = "./aws.yaml") -> None:
    """
    Download and unzip image from URL. The URL is set in the config yaml file with key: 
    image-zip:
        url: ...

    Args:
        cached_path (str, optional): The path of the cached folder. Defaults to "./data/".
        config_file (str, optional): The path of the config file. Defaults to "./aws.yaml".
        
    """
    if os.path.exists(cached_path + "images/"):
        print("Images data already existed")
        return 
    
    os.makedirs(cached_path, exist_ok=True)

    image_url = read_cloud_config(config_file)["image-zip"]["url"]
    
    resp = requests.get(image_url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    
    file_name = image_url.split("/")[-1]
    
    with open(cached_path+file_name, 'wb') as file, tqdm(
        desc=file_name,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
            
    with ZipFile(cached_path+file_name, 'r') as zipObject:
        zipObject.extractall(cached_path + "images/")
        
    print(f"{len(fnmatch.filter(os.listdir(cached_path), '*.jpg'))} images downloaded and unzipped to folder {cached_path}")