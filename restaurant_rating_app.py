# restaurant_rating_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# -------------------------------
# 1. LOAD DATA
# -------------------------------
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
    
    # Drop rows with missing target or key features
    df = df.dropna(subset=['rate (out of 5)', 'avg cost (two people)', 'num of ratings'])
    
    # Feature engineering
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
    }).fillna(1)
    
    return df

df = load_data()

# -------------------------------
# 2. FEATURE SELECTION
# -------------------------------
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

# -------------------------------
# 3. SCALING
# -------------------------------
scaler = StandardScaler()
# Fit scaler on numeric columns only
numeric_cols = ['num of ratings', 'avg cost (two people)']
X_scaled = X.copy()
X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# -------------------------------
# 4. MODEL TRAINING
# -------------------------------
model = LinearRegression()
model.fit(X_scaled, y)

# -------------------------------
# 5. STREAMLIT UI
# -------------------------------
st.title("🍽️ Restaurant Rating Prediction App")
st.write("Predict the restaurant rating (out of 5) using business features.")

st.sidebar.header("Enter Restaurant Details")

num_ratings = st.sidebar.number_input("Number of Ratings", min_value=0, value=50)
avg_cost = st.sidebar.number_input("Average Cost for Two People", min_value=0, value=500)
online_order = st.sidebar.selectbox("Online Order Available?", ['Yes', 'No'])
table_booking = st.sidebar.selectbox("Table Booking Available?", ['Yes', 'No'])
cost_category = st.sidebar.selectbox("Cost Category", ['low', 'medium', 'high', 'luxury'])

# -------------------------------
# 6. INPUT PROCESSING
# -------------------------------
# Encode Yes/No
online_order_val = 1 if online_order == 'Yes' else 0
table_booking_val = 1 if table_booking == 'Yes' else 0
cost_category_val = {'low': 0, 'medium': 1, 'high': 2, 'luxury': 3}[cost_category]

# Scale numeric inputs properly using a DataFrame to avoid warnings
input_numeric = pd.DataFrame(
    [[num_ratings, avg_cost]],
    columns=['num of ratings', 'avg cost (two people)']
)
scaled_numeric = scaler.transform(input_numeric)
num_ratings_scaled = scaled_numeric[0][0]
avg_cost_scaled = scaled_numeric[0][1]

# Calculate popularity_score based on scaled rating
# Here we use the average rating from dataset to make it realistic
avg_rating_mean = df['rate (out of 5)'].mean()
popularity_score_val = num_ratings * avg_rating_mean

# Prepare final input array
input_data = np.array([[
    num_ratings_scaled,
    avg_cost_scaled,
    popularity_score_val,
    online_order_val,
    table_booking_val,
    cost_category_val
]])

# -------------------------------
# 7. PREDICTION
# -------------------------------
if st.button("Predict Rating"):
    prediction = model.predict(input_data)[0]
    
    # Clamp between 0 and 5
    prediction = max(0, min(5, prediction))
    
    st.success(f"⭐ Predicted Restaurant Rating: **{prediction:.2f} / 5**")
