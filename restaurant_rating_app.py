# restaurant_rating_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
# 1. LOAD DATA
@st.cache_data
def load_data():
    df = pd.read_csv(r"D:\certificationtask\task\restaurant.csv")
    # Drop unnecessary columns
    df = df.drop(
        columns=['Unnamed: 0', 'Unnamed: 0.1', 'restaurant name', 'local address'],
        errors='ignore'
    )
    # Encode Yes/No columns
    df['online_order'] = df['online_order'].map({'Yes': 1, 'No': 0})
    df['table booking'] = df['table booking'].map({'Yes': 1, 'No': 0})
    # Drop rows with missing target or important features
    df = df.dropna(subset=['rate (out of 5)', 'avg cost (two people)', 'num of ratings'])
    # Feature Engineering
    df['popularity_score'] = df['num of ratings'] * df['rate (out of 5)']
    df['cost_category'] = pd.cut(
        df['avg cost (two people)'],
        bins=[0, 300, 600, 1000, 6000],
        labels=['low', 'medium', 'high', 'luxury']
    )
    df['cost_category_encoded'] = df['cost_category'].map({
        'low': 0,
        'medium': 1,
        'high': 2,
        'luxury': 3
    })
    df['cost_category_encoded'] = df['cost_category_encoded'].fillna(1)
    return df
df = load_data()
# 2. FEATURE SELECTION
features = [
    'num of ratings',
    'avg cost (two people)',
    'popularity_score',
    'online_order',
    'table booking',
    'cost_category_encoded'
]
X = df[features]
y = df['rate (out of 5)']
# 3. SCALING
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[['num of ratings', 'avg cost (two people)']] = scaler.fit_transform(
    X[['num of ratings', 'avg cost (two people)']]
)
# 4. MODEL TRAINING
model = LinearRegression()
model.fit(X_scaled, y)
# 5. STREAMLIT UI
st.title("Restaurant Rating Prediction App")
st.write("Predict the restaurant rating (out of 5) using business features.")
st.sidebar.header("Enter Restaurant Details")
num_ratings = st.sidebar.number_input(
    "Number of Ratings", min_value=0, value=50
)
avg_cost = st.sidebar.number_input(
    "Average Cost for Two People", min_value=0, value=500
)
online_order = st.sidebar.selectbox(
    "Online Order Available?", ['Yes', 'No']
)
table_booking = st.sidebar.selectbox(
    "Table Booking Available?", ['Yes', 'No']
)
cost_category = st.sidebar.selectbox(
    "Cost Category", ['low', 'medium', 'high', 'luxury']
)
# 6. INPUT PROCESSING
online_order_val = 1 if online_order == 'Yes' else 0
table_booking_val = 1 if table_booking == 'Yes' else 0
cost_category_val = {'low': 0, 'medium': 1, 'high': 2, 'luxury': 3}[cost_category]
# Approximate popularity score for prediction
popularity_score_val = num_ratings * 3.0
# Scale numeric inputs (CORRECT WAY)
scaled_values = scaler.transform([[num_ratings, avg_cost]])
num_ratings_scaled = scaled_values[0][0]
avg_cost_scaled = scaled_values[0][1]
# Prepare final input array
input_data = np.array([[
    num_ratings_scaled,
    avg_cost_scaled,
    popularity_score_val,
    online_order_val,
    table_booking_val,
    cost_category_val
]])
# 7. PREDICTION
if st.button("Predict Rating"):
    prediction = model.predict(input_data)[0]
    # Clamp output between 0 and 5
    prediction = max(0, min(5, prediction))
    st.success(f"Predicted Restaurant Rating: **{prediction:.2f} / 5**")
