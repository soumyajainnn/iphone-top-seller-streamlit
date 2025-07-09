import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load trained model
with open("iphone_classifier_final.pkl", "rb") as f:
    model = pickle.load(f)

# Page setup
st.set_page_config(page_title="iPhone Top Seller Predictor", page_icon="📱")
st.title("📱 iPhone Top Seller Predictor")
st.markdown("Predict whether an iPhone will be a **Top Seller** based on its pricing and specs.")

# Input form
with st.form("prediction_form"):
    sale_price = st.number_input("Sale Price (₹)", min_value=10000, max_value=200000, value=95000)
    mrp = st.number_input("MRP (₹)", min_value=10000, max_value=250000, value=105000)
    discount = st.number_input("Discount Percentage (%)", min_value=0.0, max_value=100.0, value=9.5)
    ram = st.selectbox("RAM (GB)", options=[2, 3, 4, 6, 8], index=3)
    submitted = st.form_submit_button("Predict")

# Prediction and chart
if submitted:
    input_data = np.array([[sale_price, mrp, discount, ram]])
    prediction = model.predict(input_data)
    result = "✅ Top Seller" if prediction[0] == 1 else "❌ Not a Top Seller"
    st.success(f"Prediction: {result}")

    # 📊 Comparison Chart
    st.subheader("📊 Your Input vs Dataset Average")

    avg_values = [85000, 95000, 11.2, 4]  # Replace with actual dataset averages if different
    user_values = [sale_price, mrp, discount, ram]
    feature_names = ["Sale Price", "MRP", "Discount (%)", "RAM"]

    x = np.arange(len(feature_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, user_values, width, label='Your Input', color='skyblue')
    bars2 = ax.bar(x + width/2, avg_values, width, label='Dataset Avg', color='lightgreen')

    # Add labels
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 2, round(height, 1), ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Values')
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names)
    ax.legend()
    ax.set_title("Feature Comparison")

    st.pyplot(fig)



