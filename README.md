# Credit Score Prediction

## Overview

This project predicts a customer's **credit score category** (Good,
Standard, or Poor) using financial and behavioral data. The goal is to
help financial institutions assess risk and automate creditworthiness
evaluation using machine learning models.

## Steps in the Project

1.  **Data Import and Cleaning** The dataset includes various customer
    details such as income, loan history, payment behavior, and monthly
    balance. The script automatically detects the target column
    (`Credit_Score`) and cleans the data by:

    -   Removing identifiers such as ID, SSN, and Name
    -   Fixing incorrect numeric values and formatting inconsistencies
    -   Converting credit history into total months
    -   Expanding `Type_of_Loan` into separate binary indicators for
        each loan type
    -   Handling placeholders and missing values

2.  **Feature Engineering** Key features include:

    -   Numeric features like income, EMI, outstanding debt, utilization
        ratio, and delayed payments
    -   Categorical features like occupation, payment behavior, and
        credit mix The script also creates derived features such as the
        number of loan types and whether a customer holds specific loan
        categories.

3.  **Data Splitting and Preprocessing** The data is divided into
    training and validation sets.

    -   Numeric features are scaled using standard scaling after median
        imputation.
    -   Categorical features are encoded using ordinal encoding after
        imputing the most frequent category. This ensures that the
        models handle mixed data types effectively.

4.  **Model Training** Two models were trained and compared:

    -   **HistGradientBoostingClassifier** --- a high-performance
        gradient boosting model well suited for tabular data
    -   **DecisionTreeClassifier** --- used as a quick baseline for
        sanity checking

5.  **Model Evaluation** On the validation data:

    **HistGradientBoosting**

    -   Accuracy: 0.7585
    -   Macro F1: 0.7424

    **DecisionTree**

    -   Accuracy: 0.6863
    -   Macro F1: 0.6774

    The HistGradientBoosting model performed better overall, achieving
    stronger balance between precision and recall across all classes.

6.  **Confusion Matrix Analysis**

    -   **HistGradientBoosting:** The model correctly predicts most
        Standard and Poor cases, with some overlap between Good and
        Standard categories.
    -   **DecisionTree:** Performs well for Poor class but tends to
        confuse Standard with Good due to simpler boundaries. These
        plots help visualize where the models make classification
        errors.

7.  **Model Selection and Final Training** Based on the higher macro F1
    score, **HistGradientBoosting** was selected as the final model. It
    was retrained on the full cleaned training dataset to maximize
    predictive power.

8.  **Predictions and Submission** The final model generated predictions
    for the test dataset. A submission file named `submission.csv` was
    created with the predicted credit score for each customer. The
    distribution of predictions was:

    -   Standard: 26,595
    -   Poor: 13,993
    -   Good: 9,412

9.  **Insights**

    -   The model effectively differentiates between customers with
        stable credit histories and those with riskier financial
        patterns.
    -   Income, payment behavior, and number of delayed payments are
        strong indicators of credit quality.
    -   Gradient boosting provides better generalization and handles
        mixed data types more efficiently than decision trees.

## How to Run the Project

1.  Place the training and test CSV files (`train.csv` and `test.csv`)
    in the same folder as `credit_score.py`.

2.  Run the script:

    ``` bash
    python credit_score.py
    ```

3.  The program will train both models, evaluate performance, display
    confusion matrices, and export predictions to `submission.csv`.
