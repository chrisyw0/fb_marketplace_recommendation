from setuptools import setup, find_packages

setup(
    name='fbmarketplace_model',
    version='0.0.1',
    python_requires=">=3.8",
    packages=find_packages(
        where='./',
        include=['fbRecommendation.dl.base.model*', 'fbRecommendation.dl.tensorflow.model*'],
        exclude=[],
    ),
    install_requires=[
        "tensorflow>=2.8",
        "tensorflow_hub",
        "gensim"
    ],
)