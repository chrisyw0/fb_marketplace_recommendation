import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
from typing import Tuple
from dataclasses import dataclass

@dataclass
class DataVisializer():
    """Visualisation the data

    Args:
        df_product (pd.DataFrame): Product dataframe
        df_image (pd.DataFrame): Image dataframe
    
    """
    df_product: pd.DataFrame
    df_image: pd.DataFrame
    
    def show_location(self, continent: str = None, country: str = None):
        """ 
        Show items' location in a map

        Args:
            continent (str, optional): The continent of the underlying map, should either enter continent or country. Defaults to None.
            country (str, optional): The country of the underlying map, should either enter continent or country. Defaults to None.
        """
        df_product_copy = self.df_product.copy()
        
        gdf = gpd.GeoDataFrame(
            df_product_copy, geometry= gpd.points_from_xy(df_product_copy.lon, df_product_copy.lat))

        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

        if continent:
            ax = world[world.continent == continent].plot(
                color='white', edgecolor='black')
        elif country:
            ax = world[world.name == country].plot(
                color='white', edgecolor='black')

        gdf.plot(ax=ax, color='red')
        plt.show()
        
    def show_root_category(self, show_unknown: bool = False):
        """
        Show items' root category distribution

        Args:
            show_unknown (bool, optional): Whether including Unknown in the graph. Defaults to False.
        """
        if not show_unknown:
            df_cat_temp = self.df_product[self.df_product["root_category"] != "Unknown"]
        else:
            df_cat_temp = self.df_product
            
        df_root_cat_count = df_cat_temp.groupby("root_category").size().sort_values()
        df_root_cat_count.plot.barh()
        
    def show_sub_categories(self, show_unknown: bool = False):
        """
        Show items' root and last category distribution

        Args:
            show_unknown (bool, optional): Whether including Unknown in the graph. Defaults to False.
        """
        if not show_unknown:
            df_cat_temp = self.df_product[self.df_product["root_category"] != "Unknown"]
        else:
            df_cat_temp = self.df_product
            
        df_sub_cat_count = df_cat_temp.groupby(["root_category", "sub_category"]).size().nlargest(10).sort_values()
        df_sub_cat_count.plot.barh()
        
    def show_price_distribution(self):
        """
        Show price distribution

        """
        self.df_product["price"].plot.box()
        
    def show_word_counts(self):
        """
        Show items' number of words distrbution

        """
        
        self.df_product[["product_name_word_count", "product_description_word_count"]].plot.box()
    
    def show_images_num(self):
        """
        Show items' number of images distrbution

        """
        
        self.df_product.groupby("image_num").size().plot.bar()
        
    def show_word_cloud(self, size: Tuple[int, int] = (800, 800), stopwords: set = None):
        """
        Show the most appearing words in the product name and description in word cloud

        Args:
            size (Tuple[int, int], optional): Size of the word cloud. Defaults to (800, 800).
            stopwords (set, optional): Words not to be included in the word cloud. Defaults to None.
        """
        
        comment_words = ''
        stopwords = set(STOPWORDS)

        custom_stop_word = {"gumtree"} if stopwords is None else stopwords
        stopwords = stopwords.union(custom_stop_word)

        for _, val in self.df_product["product_name_description"].iteritems():
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
        
    def show_images_summary(self):
        """
        Show images meta data (image width, height, width-to-height ratio, mode)
        """
        print(self.df_image.describe())
        
        _, axes = plt.subplots(nrows=2, ncols=2)
        
        self.df_image["image_width"].plot.hist(ax=axes[0,0])
        self.df_image["image_height"].plot.hist(ax=axes[0,1])
        self.df_image["image_ratio"].plot.hist(ax=axes[1,0])
        
        df_image_mode = self.df_image.groupby("image_mode").size().sort_values(ascending=False)
        df_image_mode.plot.bar(ax=axes[1,1])
        
        plt.show()
        
    def visualise_data(self):
        """
        Visualise the image and product data frame
        """
        
        self.show_location(country="United Kingdom")
        self.show_root_category()
        self.show_sub_categories()
        self.show_price_distribution()
        self.show_word_counts()
        self.show_word_cloud()
        self.show_images_num()
        
        self.show_images_summary()