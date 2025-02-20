# 📊 COPD & Asthma Diagnosis Model

This project aims to analyze the diagnosis data of COPD and asthma patients within the scope of **Statistical Data Mining** and predict the disease using a logistic regression model.

## 🗂 Data Preprocessing

The first step in building a successful model in data mining is data preprocessing. The following steps have been performed:

### 📥 Data Loading and Initial Examination
First, the dataset was read using pandas, and the variable types were examined.

```python
import pandas as pd
import numpy as np

file_path = 'orjinal_veri.xlsx'
data = pd.read_excel(file_path)

# Information about the dataset
print(data.info())
print(data.dtypes)
```
**Result:** Some variables in the dataset contain missing values, and some variables are incorrectly defined in terms of type.

### 🗑️ Removing Irrelevant Variables
Variables that do not contribute to the analysis have been removed from the dataset.

```python
data = data.drop(['varsakımde ANNE', 'varsakımde BABA', 'varsakımde KARDES', 'varsakımde DİĞER'], axis=1)
data = data.drop(['yogumbakımatoplamyatıssuresısaat', 'servıseoplamyatıssuresısaat'], axis=1)
```
**Result:** These variables were determined to be irrelevant to the model and were removed.

### 📏 Adding New Variables
A new variable, **BMI (Body Mass Index)**, was added to the dataset.

```python
data['boy'] = data['boy'] / 100
data['VKİ'] = data['vucutagırlıgı'] / (data['boy'] ** 2)
data = data.drop(['boy', 'vucutagırlıgı'], axis=1)
```
**Result:** The BMI variable was created to analyze individuals' physical conditions.

### 🏷️ Converting Categorical Variables
Categorical variables were converted to the **category** type, and value labels were assigned.

```python
data['cınsıyet'] = data['cınsıyet'].replace({1: "MALE", 2: "FEMALE"}).astype('category')
data['sıgarakullanımı'] = data['sıgarakullanımı'].replace({1: "NEVER SMOKED", 2: "QUIT", 3: "CURRENTLY SMOKING"}).astype('category')
```
**Result:** This transformation helped the model better interpret categorical variables.

### ❓ Handling Missing Values
Missing values were checked and filled using appropriate methods.

```python
data['kanbasıncıdıastolık'].fillna(data['kanbasıncıdıastolık'].median(), inplace=True)
data['kanbasıncısıstolık'].fillna(data['kanbasıncısıstolık'].median(), inplace=True)
```
**Result:** Filling missing values with the median improved the statistical reliability of the dataset.

### 📊 Modeling and Evaluation

The model's performance was analyzed using **confusion matrix, accuracy score, and classification report**.

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
```
**Result:**
- **Accuracy:** 94% for the training set and 89% for the test set.
- **Confusion Matrix:** The model's error rate is observed to be low.

### 📈 Model Assumptions and Regression Coefficients
Logistic regression model coefficients and **Odds Ratios** were calculated.

```python
import statsmodels.api as sm
X_const = sm.add_constant(X)
logit_model = sm.Logit(y, X_const)
result = logit_model.fit()
print(result.summary())
```
**Result:** The independent variables explain 75% (R² = 0.75) of the dependent variable.

## 🎯 Conclusion
In this project, COPD and asthma diagnosis data were processed, and disease predictions were made using a logistic regression model. **An accuracy rate of 89%** was achieved. Alternative algorithms can be explored to improve the model.

---
