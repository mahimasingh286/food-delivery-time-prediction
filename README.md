# food-delivery-time-prediction
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv(r"C:\Users\immah\Downloads\food_delivery_time_prediction.csv")
print("Initial DataFrame:")
print(df)


df.fillna(method="ffill", inplace=True)


df["delivery_time"] = pd.to_numeric(df["delivery_time"], errors='coerce')


if df["delivery_time"].isnull().any():
    print("Warning: NaN values found in 'delivery_time'. Dropping these rows.")
    df.dropna(subset=["delivery_time"], inplace=True)  


df["is_late"] = (df["delivery_time"] > 10).astype(int)


feature = ["Weatherconditions", "Road_traffic_density", "Vehicle_condition", "Type_of_order", "Type_of_vehicle"]
x = df[feature]

label_encoders = {}
for col in feature:
    le = LabelEncoder()
    x[col] = le.fit_transform(x[col].astype(str))  
    label_encoders[col] = le


y_time = df["delivery_time"]

if y_time.isnull().any():
    print("Warning: NaN values found in target variable 'y_time'. Dropping NaN values.")
    y_time = y_time.dropna() 
x = x.loc[y_time.index]  

if x.empty or y_time.empty:
    print("Error: The resulting train set will be empty. Please check your data.")
else:
   
    x_train, x_test, y_time_train, y_time_test = train_test_split(x, y_time, test_size=0.2, random_state=42)

  
    lr_model = LinearRegression()
    lr_model.fit(x_train, y_time_train)

    
    y_time_pred = lr_model.predict(x_test)

    
    if np.any(np.isnan(y_time_pred)):
        print("Warning: Predictions contain NaN values.")

    
    y_time_pred_str = [str(time) for time in y_time_pred]

    
    print("Predicted delivery times (as strings):")
    print(y_time_pred_str)

    
    mae = mean_absolute_error(y_time_test, y_time_pred)
    print(f"Mean Absolute Error: {mae}")

   
    y_time_str = [str(time) for time in y_time_test]

    
    print("Original delivery times (as strings):")
    print(y_time_str)

    
    y_late = df["is_late"]

   
    X_train_class, X_test_class, y_late_train, y_late_test = train_test_split(x, y_late, test_size=0.2, random_state=42)

    
    log_model = LogisticRegression()
    log_model.fit(X_train_class, y_late_train)

    
    y_late_pred = log_model.predict(X_test_class)

   
    print("\n📦 Logistic Regression Results:")
    print("Accuracy:", accuracy_score(y_late_test, y_late_pred))
    print(classification_report(y_late_test, y_late_pred))
