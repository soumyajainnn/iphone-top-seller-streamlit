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

# Prediction + Charts
if submitted:
    input_data = np.array([[sale_price, mrp, discount, ram]])
    prediction = model.predict(input_data)
    result = "✅ Top Seller" if prediction[0] == 1 else "❌ Not a Top Seller"
    st.success(f"Prediction: {result}")

    # Dataset average values
    avg_sale_price = 85000
    avg_mrp = 95000
    avg_discount = 11.2
    avg_ram = 4

    # 🔹 Chart 1: Sale Price vs MRP
    st.subheader("💰 Price Comparison")

    features1 = ["Sale Price", "MRP"]
    user_values1 = [sale_price, mrp]
    avg_values1 = [avg_sale_price, avg_mrp]

    fig1, ax1 = plt.subplots()
    x1 = np.arange(len(features1))
    width = 0.35

    ax1.bar(x1 - width/2, user_values1, width, label='Your Input', color='skyblue')
    ax1.bar(x1 + width/2, avg_values1, width, label='Dataset Avg', color='lightgreen')

    ax1.set_ylabel('₹')
    ax1.set_xticks(x1)
    ax1.set_xticklabels(features1)
    ax1.set_title('Sale Price & MRP Comparison')
    ax1.legend()

    st.pyplot(fig1)

    # 🔹 Chart 2: Discount (%) vs RAM
    st.subheader("📉 Discount & RAM Comparison")

    features2 = ["Discount (%)", "RAM (GB)"]
    user_values2 = [discount, ram]
    avg_values2 = [avg_discount, avg_ram]

    fig2, ax2 = plt.subplots()
    x2 = np.arange(len(features2))

    ax2.bar(x2 - width/2, user_values2, width, label='Your Input', color='orange')
    ax2.bar(x2 + width/2, avg_values2, width, label='Dataset Avg', color='lightgreen')

    ax2.set_ylabel('Value')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(features2)
    ax2.set_title('Discount & RAM Comparison')
    ax2.legend()

    st.pyplot(fig2)

# Add a divider and footer credit
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Made by <strong>Soumya Jain</strong></p>",
    unsafe_allow_html=True
)





