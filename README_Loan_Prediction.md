# Loan Prediction Using Machine Learning

## Project Overview

This project explores a loan application dataset and builds a machine
learning classification model to predict the loan-status label from
applicant information. The notebook uses **Logistic Regression** as its
main model.

## Objective

Use applicant and loan-related features to predict the `Loan_Status`
target and evaluate the model's performance on a held-out test set.

## Dataset

The notebook loads the dataset from `data/loan_data.csv`.

The columns used in the notebook include:

-   `Gender`
-   `Married`
-   `Dependents`
-   `Education`
-   `Self_Employed`
-   `ApplicantIncome`
-   `CoapplicantIncome`
-   `LoanAmount`
-   `Loan_Amount_Term`
-   `Credit_History`
-   `Property_Area`
-   `Loan_Status` --- target label
-   `Loan_ID` --- removed before model training

Make sure `loan_data.csv` is placed in a folder named `data` relative to
the notebook, or update the file path in the notebook.

## Tools and Libraries

-   Python
-   Pandas and NumPy
-   Matplotlib
-   Scikit-learn

## Workflow

1.  Load the dataset.
2.  Inspect data types, summary statistics, and missing values.
3.  Fill selected missing values using the mode or median.
4.  Separate the features from the target and drop `Loan_ID`.
5.  Convert categorical values to numeric values.
6.  Split the data into training and testing sets using an 80/20 split
    and stratification.
7.  Standardize selected numeric columns using `StandardScaler`.
8.  Train a Logistic Regression classifier.
9.  Evaluate the model using accuracy, a classification report, and a
    confusion matrix.
10. Explore feature selection with `SelectKBest` and the chi-squared
    test.
11. Try predictions on new applicant examples.

## Model Evaluation

The notebook reports: - Accuracy - Precision, recall, and F1-score -
Confusion matrix

Run the notebook to obtain the current results for your local copy of
the dataset.

## Important Note About Target Labels

In the notebook, `Loan_Status` is mapped as `Y = 0` and `N = 1`.
However, the example prediction code prints `YES` when the model
predicts `1` and `NO` otherwise. That display logic is inconsistent with
the mapping and should be corrected before relying on the example
predictions. With the current mapping, `0` corresponds to `Y` and `1`
corresponds to `N`.

Also, the notebook's sample `new_data` is passed to the model without
applying the same scaling transformation used for the training data. For
reliable predictions, apply the fitted scaler to the corresponding
numeric input columns using the same preprocessing steps as training.

## How to Run

1.  Clone or download this repository.

2.  Install the required libraries:

    ``` bash
    pip install pandas numpy matplotlib scikit-learn jupyter
    ```

3.  Place `loan_data.csv` in the `data/` folder.

4.  Open and run `Loan_prediction.ipynb` in Jupyter Notebook or
    JupyterLab.

## Project Status

Educational machine learning project for practicing data cleaning,
preprocessing, classification, and model evaluation.

## Disclaimer

This notebook is for learning and experimentation. Its predictions
should not be used as the sole basis for real lending decisions.
