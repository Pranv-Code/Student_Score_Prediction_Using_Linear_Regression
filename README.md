# Student Performance Prediction with Linear Regression 🎓

This project uses **Linear Regression** from `scikit-learn` to build a predictive model for a student's **Final Score** based on several key input features.

---

## ⚙️ Installation

To run this project, you need **Python 3.x** and the following libraries. You can install them using `pip`:

```bash
pip install pandas scikit-learn
```
## 🚀 How to Run
Dataset: Ensure data file ('student_dataset.csv') in the same directory.

Execute: Run the Python script (e.g., score_prediction_LE.py):
``` bash    
python score_prediction_LE.py
```
## 📊 Program Overview
The script performs the following steps:

Data Loading: Reads the student_dataset.csv.

Data Splitting: Splits the features (Age, Study_hours_per_day, etc.) and the target (Final_Score) into training (20%) and testing (80%) sets.

Model Training: Trains a LinearRegression model on the training data.

Prediction & Evaluation: Makes predictions on the test set and calculates key performance metrics.

New Prediction: Uses the trained model to predict the final score for a sample student.

## 📈 Performance Metrics
The model's performance on the test data is evaluated using the following metrics:
MAE (Mean Absolute Error)

MSE (Mean Squared Error)

RMSE (Root Mean Squared Error)

R² (Coefficient of Determination) - A value closer to 1 indicates a better fit.

## 🎯 Example Output
The script prints the calculated metrics and the predicted score for a new data point (a student with: Age=19, Study_hours=5, Midterm=53, etc.).

MAE :  2.53

MSE :  10.43

RMSE :  3.23

R^2 :  0.93

Predicted Final Score based on trained model :  69.72#
