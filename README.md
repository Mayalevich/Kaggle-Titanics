# Kaggle-Titanics
## Titanic Data Analysis and Model Training
This repository contains a Jupyter notebook for analyzing the Titanic dataset, performing data preprocessing, and training a RandomForestClassifier to predict survival on the Titanic. The notebook includes data visualization, feature engineering, and model evaluation.

## Prerequisites

- Python 3.x
- Jupyter Notebook
- Required Python libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn


## Dataset

The dataset used in this notebook is the Titanic dataset, which can be found on [Kaggle](https://www.kaggle.com/c/titanic/data). The dataset includes information about the passengers on the Titanic, such as age, sex, fare, and whether they survived or not.

## Notebook Structure

1. **Load and Explore Data**
   - Load the dataset using pandas
   - Display basic statistics and data types
   - Visualize correlations using a heatmap

2. **Data Preprocessing**
   - Perform stratified splitting to create training and test sets
   - Define custom transformers for imputing missing values and encoding categorical features
   - Create a preprocessing pipeline

3. **Model Training**
   - Standardize the features
   - Perform grid search for hyperparameter tuning of the RandomForestClassifier
   - Train the model using the best hyperparameters

4. **Model Evaluation**
   - Transform the test set using the preprocessing pipeline
   - Standardize the test set features
   - Evaluate the model on the test set

### AgeImputer
This transformer imputes missing values in the `Age` column using the mean strategy.

### FeatureEncoder
This transformer encodes categorical features (`Embarked` and `Sex`) using one-hot encoding.

### FeatureDropper
This transformer drops unnecessary features (`Embarked`, `Name`, `Ticket`, `Cabin`, and `Sex`).

## Results

The model was evaluated on the test set and achieved a score of 0.78468. This result placed the model in position 2403 on the Kaggle leaderboard.

## Acknowledgements

- The dataset is provided by [Kaggle](https://www.kaggle.com/c/titanic/data).
- The notebook uses Python libraries such as pandas, numpy, matplotlib, seaborn, and scikit-learn.
