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

# Prediction and Chart
if submitted:
    input_data = np.array([[sale_price, mrp, discount, ram]])
    prediction = model.predict(input_data)
    result = "✅ Top Seller" if prediction[0] == 1 else "❌ Not a Top Seller"
    st.success(f"Prediction: {result}")

    # --- Dataset Averages ---
    avg_values = {
        "Sale Price (₹)": 85000,
        "MRP (₹)": 95000,
        "Discount (%)": 11.2,
        "RAM (GB)": 4
    }

    user_inputs = {
        "Sale Price (₹)": sale_price,
        "MRP (₹)": mrp,
        "Discount (%)": discount,
        "RAM (GB)": ram
    }

    st.subheader("📊 Feature-wise Comparison with Dataset Average")

    for feature in avg_values.keys():
        fig, ax = plt.subplots()
        ax.bar(["Your Input", "Dataset Avg"], [user_inputs[feature], avg_values[feature]], 
               color=['skyblue', 'lightgreen'])
        ax.set_title(f"{feature}: Your Input vs Dataset Avg")
        ax.set_ylabel(feature)
        st.pyplot(fig)

