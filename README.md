--Diabetes Prediction Using Machine Learning--

-- Project Overview--

This project develops a machine learning system for predicting diabetes outcomes using patient health-related features.

The project includes data preprocessing, exploratory data analysis, multiple machine learning models, model comparison, hyperparameter tuning, and an interactive Streamlit application.

--Dataset--

The dataset contains 768 patient records and 8 input features:

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

Target variable:

- Outcome: 0 = No Diabetes
- Outcome: 1 = Diabetes

--Data Preprocessing--

The following preprocessing steps were performed:

- Checked data types
- Checked missing values
- Checked duplicate records
- Identified zero values representing missing-like measurements
- Replaced invalid zero values with missing values for:
  - Glucose
  - Blood Pressure
  - Skin Thickness
  - Insulin
  - BMI
- Applied median imputation
- Applied feature scaling

-- Machine Learning Models--

The following models were compared:

- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Support Vector Machine
- Gradient Boosting

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Five-fold stratified cross-validation was also performed.

-- Final Model --

Hyperparameter tuning was performed using cross-validation with ROC-AUC as the optimization metric.

The selected Logistic Regression model used:

```text
C=1
```
--Final Model Performance--

The final Logistic Regression model was evaluated on the held-out test set.

| Metric     | Score |
| Accuracy   | 70.8% |
| Precision  | 60.0% |
| Recall     | 50.0% |
| F1 Score   | 54.5% |
| ROC-AUC    | 81.3% |


--Streamlit Application--

An interactive Streamlit application was developed where users can enter patient information and receive a machine-learning prediction.

The application uses the trained model saved as:

`diabetes_model.pkl`

--Live Demo--
[Try the diabetes prediction system]
https://diabetes-prediction-ml-g6vqwahapp32ujbxsbgdbcd.streamlit.app/

