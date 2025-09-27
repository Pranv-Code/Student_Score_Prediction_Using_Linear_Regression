import pandas as pd 
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,root_mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#^ Loading Dataset 
dataset = pd.read_csv('student_dataset.csv')
print(dataset.head()) # reviewing dataset

#^ Initializing the Inputs(features) and outputs

features = dataset[['Age','Study_hours_per_day','Midterm_Score','Assignments_Avg','Participation_Score','Quizzes_Avg','Attendance (%)']]
outputs = dataset['Final_Score']

#^ Training Testing and Splitting dataset values 
X_train,X_test,Y_train,Y_test=train_test_split(features,outputs,random_state=42,train_size=0.2)

#^ Training Model 
model = LinearRegression() # using linear regression for numerical prediction (Basic)
model.fit(X_train,Y_train)

predicted_values = model.predict(X_test)

#^ Calculating the mretices for performance
mae = mean_absolute_error(Y_test,predicted_values)
mse = mean_squared_error(Y_test,predicted_values)
rmse = root_mean_squared_error(Y_test,predicted_values)
r2 = r2_score(Y_test,predicted_values)

print("MAE : ",round(mae,2))
print("MSE : ",round(mse,2))
print("RMSE : ",round(rmse,2))
print("R^2 : ",round(r2,2)) #! closer to 1 -> good <0 -> Worst


age = 19
study_hrs = 5
mid_score = 53
assignment_score = 62
parti_score = 7
quiz_score = 76
attendance = 65
new_data = pd.DataFrame(
    [[age,study_hrs,mid_score,assignment_score,parti_score,quiz_score,attendance]], 
    columns=['Age','Study_hours_per_day','Midterm_Score','Assignments_Avg','Participation_Score','Quizzes_Avg','Attendance (%)']
    )
new_predict = model.predict(new_data)[0]
print("Predicted Final Score based on trained model : ",round(new_predict,2))

