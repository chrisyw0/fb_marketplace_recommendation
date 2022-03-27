import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
from typing import Tuple

def show_location(df_product: pd.DataFrame, continent: str = None, country: str = None):
    """ 
    Show items' location in a map

    Args:
        df_product (pd.DataFrame): Product dataframe
        continent (str, optional): The continent of the underlying map, should either enter continent or country. Defaults to None.
        country (str, optional): The country of the underlying map, should either enter continent or country. Defaults to None.
    """
    gdf = gpd.GeoDataFrame(
        df_product, geometry= gpd.points_from_xy(df_product.lon, df_product.lat))

    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

    if continent:
        ax = world[world.continent == continent].plot(
            color='white', edgecolor='black')
    elif country:
        ax = world[world.name == country].plot(
            color='white', edgecolor='black')

    gdf.plot(ax=ax, color='red')
    plt.show()
    
def show_root_category(df_product: pd.DataFrame, show_unknown: bool = False):
    """
    Show items' root category distribution

    Args:
        df_product (pd.DataFrame): Product dataframe
        show_unknown (bool, optional): Whether including Unknown in the graph. Defaults to False.
    """
    if not show_unknown:
        df_cat_temp = df_product[df_product["root_category"] != "Unknown"]
    else:
        df_cat_temp = df_product
        
    df_root_cat_count = df_cat_temp.groupby("root_category").size().sort_values()
    df_root_cat_count.plot.barh()
    
def show_sub_categories(df_product: pd.DataFrame, show_unknown: bool = False):
    """
    Show items' root and last category distribution

    Args:
        df_product (pd.DataFrame): Product dataframe
        show_unknown (bool, optional): Whether including Unknown in the graph. Defaults to False.
    """
    if not show_unknown:
        df_cat_temp = df_product[df_product["root_category"] != "Unknown"]
    else:
        df_cat_temp = df_product
        
    df_sub_cat_count = df_cat_temp.groupby(["root_category", "sub_category"]).size().nlargest(10).sort_values()
    df_sub_cat_count.plot.barh()
    
def show_price_distribution(df_product: pd.DataFrame):
    """
    Show price distribution

    Args:
        df_product (pd.DataFrame): Product dataframe
    """
    df_product["price"].plot.box()
    
def show_word_counts(df_product: pd.DataFrame):
    """
    Show items' number of words distrbution

    Args:
        df_product (pd.DataFrame): Product dataframe
    """
    
    df_product[["product_name_word_count", "product_description_word_count"]].plot.box()
  
def show_images_num(df_product: pd.DataFrame):
    """
    Show items' number of images distrbution

    Args:
        df_product (pd.DataFrame): Product dataframe
    """
    
    df_product.groupby("image_num").size().plot.bar()
    
def show_word_cloud(df_product: pd.DataFrame, size: Tuple[int, int] = (800, 800), stopwords: set = None):
    """
    Show the most appearing words in the product name and description in word cloud

    Args:
        df_product (pd.DataFrame): Product dataframe
        size (Tuple[int, int], optional): Size of the word cloud. Defaults to (800, 800).
        stopwords (set, optional): Words not to be included in the word cloud. Defaults to None.
    """
    
    comment_words = ''
    stopwords = set(STOPWORDS)

    custom_stop_word = {"gumtree"} if stopwords is None else stopwords
    stopwords = stopwords.union(custom_stop_word)

    for index, val in df_product["product_name_description"].iteritems():
        val = str(val)
        tokens = val.split()
        
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()
        
        comment_words += " ".join(tokens)+" "

    wordcloud = WordCloud(width = size[0], height = size[1],
                    background_color ='white',
                    stopwords = stopwords,
                    min_font_size = 10).generate(comment_words)

    # plot the WordCloud image					
    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)

    plt.show()
    
def show_images_summary(df_image: pd.DataFrame):
    """
    Show images meta data (image width, height, width-to-height ratio, mode)

    Args:
        df_image (pd.DataFrame): Image dataframe
    """
    print(df_image.describe())
    
    _, axes = plt.subplots(nrows=2, ncols=2)
    
    df_image["image_width"].plot.hist(ax=axes[0,0])
    df_image["image_height"].plot.hist(ax=axes[0,1])
    df_image["image_ratio"].plot.hist(ax=axes[1,0])
    
    df_image_mode = df_image.groupby("image_mode").size().sort_values(ascending=False)
    df_image_mode.plot.bar(ax=axes[1,1])
    
    plt.show()
    
def visualise_data(df_product: pd.DataFrame, df_image: pd.DataFrame):
    """
    Visualise the image and product data frame

    Args:
        df_product (pd.DataFrame): Product dataframe
        df_image (pd.DataFrame): Image dataframe
    """
    
    show_location(df_product, country="United Kingdom")
    show_root_category(df_product)
    show_sub_categories(df_product)
    show_price_distribution(df_product)
    show_word_counts(df_product)
    show_word_cloud(df_product)
    show_images_num(df_product)
    
    show_images_summary(df_image)