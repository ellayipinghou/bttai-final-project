# Income Classification – ML Foundations Final Project 💼

This repository contains my final project for the Machine Learning Foundations course offered by eCornell, as part of the Break Through Tech AI Program at MIT. 
The goal of the project is to develop a supervised machine learning model that predicts whether a person earns more than $50K per year based on demographic and work-related features.

## Project Overview 📌
Using the [Adult Census Income Dataset](https://www.kaggle.com/datasets/uciml/adult-census-income), I built a binary classification model to determine whether an individual earns: `>50K` or `<=50K`. 
The project demonstrates a complete ML pipeline, including data cleaning, feature engineering, model training, evaluation, and analysis.

💡**Project Motivation:**
Predicting income can support a variety of real-world business applications, such as:

- Customer segmentation: Grouping users by likely income range for better personalization
- Targeted marketing: Recommending or advertising products & service plans that align with a customer’s budget
- Pricing strategy: Adjusting offers to improve affordability and conversion rates
- Credit/lending decisions: Estimating income to inform risk assessments (though many companies collect this data directly via applications)

By inferring income from available demographic and behavioral data, businesses can better understand their users and improve strategic decision-making. However, it’s also important to consider the ethical 
implications of such predictions, particularly in contexts like lending or insurance, to ensure responsible and fair use of AI.

🧠 **Key Concepts Demonstrated:**
- Supervised Learning & Binary Classification
- Data Preprocessing: Handling missing values, winsorization, one-hot encoding, train-test split
- Data exploration and Visualization
- Building Tree-Based Models: Simple decision trees, random forests, gradient-boosted decision trees
- Feature Selection & Model Selection: Hyperparameter tuning, grid search with cross-validation, threshold tuning
- Model Evaluation: Accuracy, AUC, precision, recall, F1 score, confusion matrix

📊 **Chosen Features:** age, education, occupation, marital-status, relationship, race, sex, hours-per-week, and native-country

⚙️ **Technologies Used:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Jupyter Notebook

## Model Performance 📈
- Accuracy: 83.8%
- AUC: 89.7%
- Precision: 65%
- Recall: 70.8%
- F1 Score: 67.8%

⚠️ **Note on Class Imbalance**: 

The dataset is imbalanced, with 76% of individuals earning <=50K. Despite using techniques like class_weight='balanced' and resampling, model performance remains uneven across classes:
- Accuracy for >50K (minority) class: 70.8%
- Accuracy for ≤50K (majority) class: 87.9%

Further tuning the classification threshold to maximize the F1 score does achieve a more balanced accuracy between the two classes, but at the expense of lower overall accuracy.
  
🧑🏽‍⚖️ **Note on Fairness Across Demographics**:

The model performs relatively well and consistently across different demographic groups, including protected attributes such as `sex`, `race`, and `native country`, in terms of overall accuracy and AUC.
However, precision and recall vary across subgroups, especially those with fewer examples in the dataset. 

This inconsistency highlights the risk of unfair outcomes for marginalized or underrepresented populations. To promote fair and equitable AI, further work is needed to collect more representative data and conduct granular fairness evaluations before deploying the model in real-world applications.
