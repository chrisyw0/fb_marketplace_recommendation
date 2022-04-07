from classes.data_preparation.download_data import DataDownloader
from classes.data_preparation.visualize_data import DataVisializer
from classes.data_preparation.clean_images import ImageCleaner
from classes.data_preparation.clean_tabular import TablularDataCleaner
from classes.ml_approach.ml_method import MachineLearningPredictor


def main():
    downloader = DataDownloader()
    
    df_product, df_image = downloader.download_data()
    downloader.download_images()
    
    image_cleaner = ImageCleaner(df_image, df_product)
    df_image_clean = image_cleaner.get_clean_image_data()
    
    product_cleaner = TablularDataCleaner(df_product, df_image_clean)
    df_product_clean = product_cleaner.get_clean_product_data()
    
    visualiser = DataVisializer(df_product_clean, df_image_clean)
    visualiser.visualise_data()
    
    ml_model_predictor = MachineLearningPredictor(df_product_clean, df_image_clean)
    
    ml_model_predictor.predict_price()
    ml_model_predictor.predict_product_type()
    

if __name__ == "__main__":
    main()