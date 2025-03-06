import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import time



import warnings
warnings.filterwarnings('ignore')

st.write("# Personal Fitness Tracker ..!!")
#st.image("", use_column_width=True)
st.write("In this WebApp you will be able to observe your predicted calories burned in your body. Pass your parameters such as `Age`, `Gender`, `BMI`, etc., into this WebApp and then you will see the predicted value of kilocalories burned.")

st.sidebar.header("User Input Parameters: ")

def user_input_features():
    age = st.sidebar.slider("Age: ", 10, 100, 30)
    bmi = st.sidebar.slider("BMI: ", 15, 40, 20)
    duration = st.sidebar.slider("Duration (min): ", 0, 35, 15)
    heart_rate = st.sidebar.slider("Heart Rate: ", 60, 130, 80)
    body_temp = st.sidebar.slider("Body Temperature (C): ", 36, 42, 38)
    gender_button = st.sidebar.radio("Gender: ", ("Male", "Female"))

    gender = 1 if gender_button == "Male" else 0

    # Use column names to match the training data
    data_model = {
        "Age": age,
        "BMI": bmi,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender_male": gender  # Gender is encoded as 1 for male, 0 for female
    }

    features = pd.DataFrame(data_model, index=[0])
    return features

df = user_input_features()

st.write("---")
st.header("Your Parameters: ")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)
st.write(df)

# Load and preprocess data
calories = pd.read_csv("calories_data.csv")

calories_train, calories_test = train_test_split(calories, test_size=0.2, random_state=1)

# Add BMI column to both training and test sets
for data in [calories_train, calories_test]:
    data["BMI"] = data["weight"] / ((data["height"] / 100) ** 2)
    data["BMI"] = round(data["BMI"], 2)

# Prepare the training and testing sets
calories_train = calories_train[["gender", "age", "BMI", "duration", "heart_rate", "body_temp", "calories"]]
calories_test = calories_test[["gender", "age", "BMI", "duration", "heart_rate", "body_temp", "calories"]]
calories_train = pd.get_dummies(calories_train, drop_first=True)
calories_test = pd.get_dummies(calories_test, drop_first=True)

# Separate features and labels
X_train = calories_train.drop("calories", axis=1)
y_train = calories_train["calories"]

X_test = calories_test.drop("calories", axis=1)
y_test = calories_test["calories"]

# Train the model
random_reg = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6)
random_reg.fit(X_train, y_train)

# Align prediction data columns with training data
df = df.reindex(columns=X_train.columns, fill_value=0)

# Make prediction
prediction = random_reg.predict(df)

st.write("---")
st.header("Prediction: ")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)

st.write(f"{round(prediction[0], 2)} **kilocalories**")

st.write("---")
st.header("Similar Results: ")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)

# Find similar results based on predicted calories
calorie_range = [prediction[0] - 10, prediction[0] + 10]
similar_data = calories[(calories["calories"] >= calorie_range[0]) & (calories["calories"] <= calorie_range[1])]
st.write(similar_data.sample(5))

st.write("---")
st.header("General Information: ")

# Boolean logic for age, duration, etc., compared to the user's input
boolean_age = (calories["age"] < df["age"].values[0]).tolist()
boolean_duration = (calories["duration"] < df["duration"].values[0]).tolist()
boolean_body_temp = (calories["body_temp"] < df["body_temp"].values[0]).tolist()
boolean_heart_rate = (calories["heart_rate"] < df["heart_rate"].values[0]).tolist()


st.write("You are older than", round(sum(boolean_age) / len(boolean_age), 2) * 100, "% of other people.")
st.write("Your exercise duration is higher than", round(sum(boolean_duration) / len(boolean_duration), 2) * 100, "% of other people.")
st.write("You have a higher heart rate than", round(sum(boolean_heart_rate) / len(boolean_heart_rate), 2) * 100, "% of other people during exercise.")
st.write("You have a higher body temperature than", round(sum(boolean_body_temp) / len(boolean_body_temp), 2) * 100, "% of other people during exercise.")