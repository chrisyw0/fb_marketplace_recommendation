from classes.data_preparation.download_data import DataDownloader
from classes.data_preparation.visualize_data import DataVisializer
from classes.data_preparation.clean_images import ImageCleaner
from classes.data_preparation.clean_tabular import TabularDataCleaner
from classes.ml_approach.ml_method import MachineLearningPredictor
from classes.cnn_approach.tensorflow.tf_image_classifier import TFImageClassifier
from classes.cnn_approach.tensorflow.tf_text_classifier import TFTextClassifier
from classes.cnn_approach.tensorflow.tf_text_classifier_transformer import TFTextTransformerClassifier
from classes.cnn_approach.tensorflow.tf_combine_classifier import TFImageTextClassifier


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

    # milestone 2: Create machine learning models to price prediction and
    # image category classification
    ml_model_predictor = MachineLearningPredictor(df_product_clean, df_image_clean)
    
    ml_model_predictor.predict_price()
    ml_model_predictor.predict_product_type()

    # milestone 3a: Create a image CNN model for category classification (RestNet50)
    image_model = TFImageClassifier(df_product=df_product_clean, df_image=df_image_clean)
    image_model.process()

    # milestone 3b: Create a image CNN model for category classification (EfficientNet)
    image_model_2 = TFImageClassifier(df_product=df_product_clean, df_image=df_image_clean)
    image_model_2.image_base_model = "EfficientNetB3"
    image_model_2.epoch = 8
    image_model_2.process()

    # milestone 4a: Create a text CNN model for category classification (Word2Vec)
    text_model = TFTextClassifier(df_product=df_product_clean, df_image=df_image_clean)
    text_model.process()

    # milestone 4b: Create a text CNN model for category classification (BERT)
    text_model_transformer = TFTextTransformerClassifier(df_product=df_product_clean, df_image=df_image_clean)
    text_model_transformer.process()

    # milestone 5a: Combining image and text model for category classification (Word2Vec)
    combine_model = TFImageTextClassifier(
        df_product=df_product_clean,
        df_image=df_image_clean,
        image_seq_layers=image_model_2.image_seq_layers,
        text_seq_layers=text_model.text_seq_layers,
        embedding_model=text_model.embedding_model
    )

    combine_model.process()

    # milestone 5b: Combining image and text model for category classification (BERT)
    combine_model_transformer = TFImageTextClassifier(
        df_product=df_product_clean,
        df_image=df_image_clean,
        image_seq_layers=image_model_2.image_seq_layers,
        text_seq_layers=text_model_transformer.text_seq_layer,
        embedding_model=text_model_transformer.embedding_model
    )

    combine_model_transformer.embedding = "BERT"
    combine_model_transformer.process()


if __name__ == "__main__":
    main()
