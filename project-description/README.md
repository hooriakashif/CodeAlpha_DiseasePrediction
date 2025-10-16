# CodeAlpha_DiseasePrediction

This repository contains the implementation of a Disease Prediction model as part of the CodeAlpha Machine Learning Internship.

## Project Overview
The project aims to predict the possibility of heart disease based on patient medical data using machine learning techniques. The implemented models include Logistic Regression, Random Forest, and SVM.

## Features
- Data loading and preprocessing
- Exploratory Data Analysis (EDA)
- Model training with multiple algorithms
- Model evaluation using accuracy, classification report, and confusion matrix
- Prediction capability with a saved model

## Files
- `data/heart.csv`: Dataset used for training
- `src/data_load.py`: Script to load and prepare the dataset
- `src/eda.py`: Script for exploratory data analysis
- `src/model_train.py`: Script to train and save the models
- `src/predict.py`: Script to make predictions using the saved model
- `models/`: Directory containing saved models and scaler

## Instructions to Run
1. Ensure all dependencies are installed: `pip install pandas numpy scikit-learn matplotlib seaborn joblib`
2. Place the `heart.csv` file in the `data` folder
3. Run the scripts in this order:
   - `python src/data_load.py`
   - `python src/eda.py`
   - `python src/model_train.py`
   - `python src/predict.py` (for testing predictions)

## Results
- Logistic Regression Accuracy: 0.7951
- Random Forest Accuracy: 0.9854
- SVM Accuracy: 0.8878
- Best model (Random Forest) saved as `models/trained_model.pkl`

## Submission
This project fulfills Task 4: Disease Prediction from Medical Data.