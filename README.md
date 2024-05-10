# Toxic-Comment-Classification

This project is part of a competition to develop a multi-headed model capable of detecting various types of toxicity in online conversations more accurately than existing models. It is designed to improve the quality of discussions on digital platforms by identifying and classifying toxic comments which may deter meaningful discourse.

## Project Description

Discussing important topics online can be challenging due to the threat of abuse and harassment, which can deter people from expressing themselves or exploring different viewpoints. The Conversation AI team, a research initiative by Jigsaw and Google, is dedicated to creating tools that enhance the quality of online conversations. This project builds on their work by aiming to refine the detection of negative online behaviors, such as threats, obscenity, insults, and identity-based hate, through advanced machine learning models.

## Solution Features

- **Text Normalization and Preprocessing:** Utilizing spaCy for robust text cleaning and preparation.
- **Exploratory Data Analysis:** Employing bar charts and correlation matrices to understand data characteristics and distributions.
- **Machine Learning Model Training:** Leveraging scikit-learn for training multiple models.
- **Hyperparameter Optimization:** Using Bayesian optimization to fine-tune model parameters for optimal performance.
- **Deep Learning Implementation:** Applying TensorFlow and Keras to develop and train advanced model architectures.
- **Performance Evaluation:** Assessing models using metrics such as accuracy, F1 score, precision, and recall.

## Disclaimer

The dataset used in this competition contains text that may be considered profane, vulgar, or offensive.

## Installation

Before running this project, ensure you have the following libraries installed:

- Python 3.8+
- pandas
- numpy
- scikit-learn
- spaCy
- TensorFlow
- keras
- matplotlib
- seaborn
- sentence_transformers
- scikit-optimize

You can install the necessary libraries using pip:

```bash
pip install numpy pandas scikit-learn spacy tensorflow keras matplotlib seaborn sentence_transformers scikit-optimize

python -m spacy download en_core_web_sm

## Usage

Load the dataset and preprocess the data using the normalization functions.
Conduct exploratory data analysis to understand the underlying patterns.
Configure and train the machine learning and deep learning models.
Evaluate the models using the provided metrics and adjust parameters as necessary.

