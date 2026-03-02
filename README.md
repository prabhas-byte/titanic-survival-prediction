# titanic-survival-prediction
Titanic Survival Prediction: A beginner machine learning project using the classic Titanic dataset to predict passenger survival with Random Forest.

# Titanic Survival Prediction

![Titanic](https://img.shields.io/badge/Kaggle-Titanic-orange?style=flat&logo=kaggle)  
A beginner-friendly machine learning project to predict whether a passenger survived the Titanic disaster using the classic Titanic dataset.

## Project Overview

The goal of this project is to build a classification model that predicts passenger survival (0 = No, 1 = Yes) based on features like age, gender, ticket class, fare, number of siblings/spouses, etc.

### Dataset
- Source: Kaggle Titanic Dataset
- Rows: 891 passengers
- Target: `Survived` (binary: 0 = did not survive, 1 = survived)

## Technologies Used
- Python 3
- pandas
- numpy
- matplotlib & seaborn (for visualization)
- scikit-learn (RandomForestClassifier)

## Key Steps Performed
- Exploratory Data Analysis (EDA)
- Handling missing values (Age, Embarked)
- Feature encoding (Sex, Embarked)
- Dropping irrelevant columns (Name, Ticket, Cabin, PassengerId)
- Train-test split (80-20)
- Model training using Random Forest Classifier
- Model evaluation using accuracy, precision, recall, F1-score, confusion matrix & feature importance

## Results
- Model Accuracy: ~81–83% (typical range with basic preprocessing)
- Most important features: Sex, Pclass, Fare, Age

## Visualizations Included
- Survival count
- Survival rate by gender
- Survival rate by passenger class
- Age distribution
- Confusion matrix
- Feature importance bar chart

## How to Run Locally

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/titanic-survival-prediction.git
   cd titanic-survival-prediction

## Install dependencies

- pip install pandas numpy matplotlib seaborn scikit-learn

## Run the Script 

- python filename.py

## License
 MIT License – feel free to use, modify, and learn from this project.

 Made with ❤️ by Prabhas
