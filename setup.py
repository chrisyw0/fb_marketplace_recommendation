from setuptools import setup, find_packages

setup(
    name='fbmarketplace_model',
    version='0.0.1',
    python_requires=">=3.8",
    package_dir={"": "fbRecommendation"},
    packages=find_packages(
        where='./fbRecommendation',
        include=['dl.*', 'dataset'],
        exclude=['dl.pytorch*'],
    ),
    install_requires=[
        "numpy",
        "pandas",
        "spacy",
        "tqdm",
        "pillow",
        "scikit-learn",
        "tensorflow",
        "tensorflow-text",
        "tensorflow_addons",
        "tf-models-official",
        "tensorboard",
        "gensim",
        "sklearn",
        "matplotlib",
        "gin-config",
        "emoji"
    ],
)