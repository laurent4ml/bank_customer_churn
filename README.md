# Predict Customer Churn

This full end to end machine learning project coded in python,
trains two models (LogisticRegression and RandomForestClassifier) using a Bank data as input
to predict customer Churn.

## Project Description
The ml project steps are:
- create and clean needed image directories
- import data from bank data set
- perform exploratory data analysis and plot visualizations
- perform feature engineering
- generate Receiver Operating Characteristic plot
- train models
- store models in local directory
- create a SHAP summary plot
- generate a Feature Importance plot
- generate a classification report for both models

## Files and data description

### Main Files
- README.md
- config.yaml
- customer_churn.py
- customer_churn_test.py
- requirements_py3.8.txt

### Test Data
- test_models/logistic_model.pkl
- test_models/rfc_model.pkl

### Generated Files
- models/logistic_model.pkl
- models/rfc_model.pkl
- logs/customer_churn.log
- logs/customer_churn_test.log
- images/results inlcudes all results images
- images/reports/logistic_regression.png
- images/reports/random_forest.png
- images/eda includes an image representing a feature graph for each feature used to train the models

### Data files
- data/bank_data.csv
The input data is store in a csv file located in the "data" directory.
It contains 10127 samples and 23 columns.
The "Attrition_Flag" column is used to set user churn.

## Running Files
To run the main script
```
python customer_churn.py
```

To run the unit tests:
```
pytest
```



