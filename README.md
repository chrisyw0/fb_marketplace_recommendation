# Facebook MarketPlace Search Ranking

This project aims at recommending buyers to a product by using a multimodel model trained by image and text datasets. Here are a few keynotes about the project: 

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
- scikit-learn

## Milestone 1

The first step of this project is to import and process the dataset. Two datasets are used in this project:
1. Product dataset
2. Image dataset

The product dataset includes products' information (name, description, price, location...) of the listed products captured from Gumtree. It has 8,091 rows and 9 columns. Images dataset includes the images information for the products. It has 12,604 rows and 5 columns. Both datasets are stored in AWS RDS (Postgres DB) which is maintained by AiCore. 

In addition, the images are zipped and stored in AWS S3. It contains all the image data (in jpg format) is available through HTTPS. 

The second step is to perform feature engineering and data cleaning. It includes generating features from existing features. For example, latitude and longitude are retrieved from the location of each product item, and image width and height for each image. Outliners were also removed in this stage, such as products with an unreasonably high price. 

After feature engineering and data cleaning, the product dataset contains 6,902 rows and 23 columns and the image dataset contains 12,604 rows and 10 columns. 

To have some insights into the dataset, I create some plots showing the distribution of the data such as price distribution, product location ..., more details can be found in the comment of the corresponding class. 
