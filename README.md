# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary Python libraries and load the dataset into the program for analysis.
2.Separate the input features and output values from the dataset for model training.
3.Split the dataset into training data and testing data to train and evaluate the model properly.
4.Train the regression model using the SGD Regressor algorithm with the training dataset.
5.Predict the output values using the testing dataset and evaluate the model performance using accuracy measures like MSE, MAE, and R² score.
## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by:DHARSHINI.M
RegisterNumber:212225220025 
*/
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
data = {
    'Area': [1500, 1800, 2400, 3000, 3500, 4000, 4200, 5000],
    'Bedrooms': [3, 4, 3, 5, 4, 6, 5, 7],
    'Age': [10, 15, 20, 8, 12, 5, 7, 3],
    'Price': [300000, 400000, 500000, 600000, 650000, 700000, 720000, 800000]
}
df = pd.DataFrame(data)
X = df[['Area', 'Bedrooms', 'Age']]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Predicted Prices:", y_pred)
print("Mean Squared Error:", mse)
print("R² Score:", r2)
```

## Output:
<img width="712" height="87" alt="image" src="https://github.com/user-attachments/assets/59f5dc31-fdd3-41bf-a9b1-0b0d38f50f80" />



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
