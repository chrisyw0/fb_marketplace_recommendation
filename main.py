from data_preparation.download_data import download_images, download_data
from data_preparation.visualize_data import visualise_data
from data_preparation.clean_images import get_clean_image_data
from data_preparation.clean_tabular import get_clean_product_data

def main():
    df_product, df_image = download_data()
    download_images()
    
    df_image_clean = get_clean_image_data(df_image, df_product)
    df_product_clean = get_clean_product_data(df_product, df_image_clean)
    
    visualise_data(df_product_clean, df_image_clean)

if __name__ == "__main__":
    main()