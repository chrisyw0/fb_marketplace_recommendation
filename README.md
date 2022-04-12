# Facebook MarketPlace Search Ranking

This project aims at recommending buyers to a product by using a multimodal model trained by image and text datasets. Here are a few keynotes about the project: 

- Processed and cleaned text and image datasets
- Designed and trained a multimodal model that combines images and text to generate embedding vectors to create a search index using Pytorch
- Developed a recommendation system based on the demographic information about potential buyers using the FAISS library from Facebook
- Created a pipeline to systematically clean new data and upsert it into a database
- Containerized the model and orchestrated the containers using Kubernetes
- Monitored and retrain the model using Kubeflow deployed on EKS 

To run the program, simply call:
```python
python main.py
```

This program use environment variables to store credentials, please set the environment according to aws.yaml in your kernel accordingly for database credentials and the image download link of the dataset. 

## Required package
- Python
- Numpy
- Pandas
- sqlalchemy
- psycopg2
- spacy
- pyyaml
- requests
- tqdm
- pillow
- wordcloud
- geopandas
- scikit-learn
- tensorflow
- keras

## Milestone 1

The first step of this project is to import and process the dataset. Two datasets are used in this project:
1. Product dataset
2. Image dataset

The product dataset includes products' information (name, description, price, location...) of the listed products captured from Gumtree. It has 8,091 rows and 9 columns. Images dataset includes the images information for the products. It has 12,604 rows and 5 columns. Both datasets are stored in AWS RDS (Postgres DB) which is maintained by AiCore. 

In addition, the images are zipped and stored in AWS S3. It contains all the image data (in jpg format) is available through HTTPS. 

The second step is to perform feature engineering and data cleaning. It includes generating features from existing features. For example, latitude and longitude are retrieved from the location of each product item, and image width and height for each image. Outliners are also removed in this stage, such as products with an unreasonably high price and images with unexpected width-to-height ratio. 

After feature engineering and data cleaning, the product dataset contains 6,902 rows and 23 columns and the image dataset contains 11,128 rows and 10 columns. 

To have some insights into the dataset, I create some plots showing the distribution of the data such as price distribution, product location ..., more details can be found in the comment of the corresponding class. 

## Milestone 2

The next step is to create simple machine learning models for predicting product price from product data and product type from image data. 

For price prediction, a linear regression model is used. The training and testing process includes:
- Generate features (one hot vector)
- Normalise data
- Split the dataset into training and testing dataset (7:3 train/test)
- Train a model with training dataset 
- Predict the price for testing dataset with the model

We have one-hot encoded the root category, coordinates from location, and tokens count from product name and description from the product dataset as features. After training with the training dataset, the model is tested with the testing dataset. Finally, the performance is then measured by RMSE. 

The RMSE for the model is around 167.6.

For product type prediction, a logistic regression model is used. After merging the product type from product dataset, 10698 images and 12 unique categories were found. 

Having similar steps as the price prediction model, we have one-hot encoded the image mode, flattened image data (10698, 144, 144, 3) -> (10698, 62208), image size (width and height) from the image dataset as features. After training with the training dataset, the model is tested with the testing dataset. Finally, the performance is then measured by accuracy.

The average accuracy for the model is around 0.14

## Milestone 3

We should never satisfy with machine learning models which gives only 14% accuracy. One possible solution is to use deep learning model. For deep leaning mode, it usually requires much more training data than machine learning model to achieve a certain level of performance. However, we only have a dataset with about 12000 images with 13 classes, which means each class has less than 1000 images. Luckily, we can use technique like transfer learning. There exists model that is well-trained with a huge dataset, which has the ability to capture the useful features in the images, and generate embeddings for the final prediction. 

To build our CNN model for category prediction, we use RestNet50 as the base model, together with data processing layer, data augmentation layer, global averaging layer, dropout layer and finally prediction layer. The input shape of the image is (256, 256, 3) and the output shape of the model is 13, which equals to the number of unique class in the dataset.   

We use the same training, validation and testing dataset as the machine model. The overall accuracy is about 55%, much better than logistic regression.  


## Reference

Deep Residual Learning for Image Recognition (CVPR 2015) (https://arxiv.org/abs/1512.03385)