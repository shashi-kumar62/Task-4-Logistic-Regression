# Task 4: Classification with Logistic Regression

## Objective
To build a binary classifier using Logistic Regression.


## Dataset
Breast Cancer Wisconsin Dataset

- Total Samples: 569
- Target Variable: diagnosis
  - M = 1 (Malignant)
  - B = 0 (Benign)


## Steps Performed

1. Loaded dataset
2. Dropped unnecessary columns (id, Unnamed: 32)
3. Converted diagnosis to binary
4. Train-test split (80-20)
5. Standardized features using StandardScaler
6. Trained Logistic Regression model
7. Evaluated using:
   - Confusion Matrix
   - Precision
   - Recall
   - ROC-AUC
8. Plotted ROC Curve
9. Performed Threshold tuning
10. Plotted Sigmoid function


## Model Results

Confusion Matrix:
[[71  1]
 [ 3 39]]

Precision: 0.975  
Recall: 0.928  
ROC-AUC Score: 0.960  

Accuracy: 96%


## Interview Questions

### 1. Difference between Logistic and Linear Regression?
Linear regression predicts continuous values.  
Logistic regression predicts probabilities for classification.

### 2. What is Sigmoid Function?
Sigmoid converts linear output into probability between 0 and 1.

### 3. What is Precision vs Recall?
Precision = TP / (TP + FP)  
Recall = TP / (TP + FN)

### 4. What is ROC-AUC?
Measures model's ability to distinguish between classes.

### 5. What is Confusion Matrix?
Table showing TP, TN, FP, FN.

### 6. What happens if classes are imbalanced?
Model may become biased towards majority class.

### 7. How do you choose threshold?
Based on business need (maximize recall or precision).

### 8. Can Logistic Regression handle multi-class?
Yes, using One-vs-Rest strategy.


## Conclusion
Logistic Regression successfully built a high-performing binary classifier with 96% accuracy and 0.96 ROC-AUC score.
