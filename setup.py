from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='A machine learning project that predicts Uber ride demand across New York City for specific future time intervals. The project follows a production-grade MLOps structure — with DVC for data versioning, Dockerized deployment, GitHub Actions for CI/CD, and a modular source layout for data ingestion, feature engineering, model training, and prediction. Built on the Cookiecutter Data Science template for clean, reproducible workflows.',
    author='Aditya Sagar',
    license='',
)
