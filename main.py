from classes.data_preparation.download_data import DataDownloader
from classes.data_preparation.visualize_data import DataVisializer
from classes.data_preparation.clean_images import ImageCleaner
from classes.data_preparation.clean_tabular import TabularDataCleaner
from classes.ml_approach.ml_method import MachineLearningPredictor
from classes.cnn_approach.image_model import ImageModel

def main():

    # milestone 1: Download and clean the data
    downloader = DataDownloader()
    
    df_product, df_image = downloader.download_data()
    downloader.download_images()
    
    image_cleaner = ImageCleaner(df_image, df_product)
    df_image_clean = image_cleaner.get_clean_image_data()
    
    product_cleaner = TabularDataCleaner(df_product, df_image_clean)
    df_product_clean = product_cleaner.get_clean_product_data()
    
    visualiser = DataVisializer(df_product_clean, df_image_clean)
    visualiser.visualise_data()

    # milestone 2: Create machine learning models for price prediction and
    # image category classification
    ml_model_predictor = MachineLearningPredictor(df_product_clean, df_image_clean)
    
    ml_model_predictor.predict_price()
    ml_model_predictor.predict_product_type()

    # milestone 3: Create CNN model for image category classification
    model = ImageModel(df_product=df_product_clean, df_image=df_image_clean)
    model.prepare_data()

    model.create_model()
    model.train_model()
    model.predict_model()
    model.visualise_performance()
    model.save_model()
    model.clean_up()

if __name__ == "__main__":
    main()
