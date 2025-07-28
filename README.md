
## 🏦Predicting Loan Default Risk with Supervised Learning
This project benchmarks supervised learning models to predict loan default risk using a structured dataset. It compares model performance using real-world evaluation metrics and highlights the impact of class imbalance and resampling techniques.


### 📂 Dataset
Source: Kaggle – Retail Bank Loan Dataset
 https://www.kaggle.com/datasets/qusaybtoush1990/machine-learning/data

Target Variable: bad_loan (1 = defaulted, 0 = paid)

Features include:

Loan grade and purpose

Income and debt-to-income ratio

Employment length and home ownership

Credit history metrics (e.g., revolving utilization)

### 🎯 Objective
To assess and compare the ability of various supervised learning models to identify default risk and support credit risk decision-making.

### 🔍 Methodology
### ✅ Data Preprocessing
Dropped redundant and high-missing-value columns

One-hot encoding for categorical features

Feature scaling with StandardScaler

Transformation of skewed variables (e.g., log scale on income and utilization)

### 📊 Exploratory Data Analysis
Distribution plots by loan grade, purpose, home ownership, and employment length

Default rate comparison across categorical segments

Correlation heatmaps for numerical features

### 🧠 Models Evaluated
Logistic Regression

K-Nearest Neighbors (KNN)

Random Forest Classifier (final model)

Each model was evaluated on both imbalanced and resampled data.

### ⚖️ Handling Class Imbalance
Applied Edited Nearest Neighbors (ENN) from imblearn to rebalance training data

Focused on improving recall and ROC-AUC for the minority class (bad_loan = 1)

### 🧪 Evaluation Metrics
Metric	Description
Accuracy	Overall correctness of the model
Precision	% of predicted bad loans that were true
Recall (Sensitivity)	% of actual bad loans correctly identified
F1-Score	Harmonic mean of precision and recall
ROC-AUC	Trade-off between sensitivity and specificity

### 🔁 Cross-Validation
Applied 5-fold cross-validation

Evaluated mean accuracy and standard deviation

### 🏁 Final Model: Tuned Random Forest on ENN Data
Metric	Value
Accuracy	78.3%
Recall (bad)	87%
Precision (bad)	69%
ROC-AUC	0.86
CV Mean Acc	~80.6%

Key predictors: grade, income, DTI, revol_util, and purpose

### 📦 Requirements
pandas  
numpy  
matplotlib  
seaborn  
scikit-learn  
imbalanced-learn  

Install with:
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn

### 🚀 How to Run
Clone this repository or download the notebook file.

Place the dataset in the same directory (loan_data.csv assumed).

Open the notebook with Jupyter.

Run all cells in order to reproduce the full workflow.
