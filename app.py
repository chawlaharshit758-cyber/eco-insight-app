
import streamlit as st
import os
import json
import hashlib
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime

# ==============================
# FILE PATHS
# ==============================
USERS_FILE = "users.json"
HISTORY_FILE = "history.csv"

# ==============================
# SAFE FILE INITIALIZATION
# ==============================
def init_files():
    # Users file
    if not os.path.exists(USERS_FILE) or os.path.getsize(USERS_FILE) == 0:
        with open(USERS_FILE, "w") as f:
            json.dump({}, f)

    # History file
    if not os.path.exists(HISTORY_FILE) or os.path.getsize(HISTORY_FILE) == 0:
        df = pd.DataFrame(columns=[
            "username", "location",
            "year", "ndvi",
            "future_year", "future_ndvi",
            "status", "timestamp"
        ])
        df.to_csv(HISTORY_FILE, index=False)

init_files()

# ==============================
# SAFE LOAD FUNCTIONS
# ==============================
def load_users():
    try:
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    except:
        return {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

def load_history():
    try:
        return pd.read_csv(HISTORY_FILE)
    except:
        return pd.DataFrame(columns=[
            "username", "location",
            "year", "ndvi",
            "future_year", "future_ndvi",
            "status", "timestamp"
        ])

def save_history(df):
    df.to_csv(HISTORY_FILE, index=False)

# ==============================
# PASSWORD SECURITY
# ==============================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# ==============================
# NDVI CALCULATION
# ==============================
def calculate_ndvi(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    red = image[:, :, 0].astype(float)
    nir = image[:, :, 1].astype(float)  # simulate NIR using green

    numerator = nir - red
    denominator = nir + red + 1e-5
    ndvi = numerator / denominator

    return float(np.mean(ndvi))

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Eco Insight Dashboard", layout="wide")

# ==============================
# SESSION STATE
# ==============================
if "user" not in st.session_state:
    st.session_state.user = None

# ==============================
# LOGIN / SIGNUP
# ==============================
if st.session_state.user is None:

    st.title("🌿 Eco-Insight Login System")

    choice = st.radio("Choose Option", ["Login", "Sign Up"])

    users = load_users()

    if choice == "Sign Up":
        new_user = st.text_input("Email")
        new_pass = st.text_input("Password", type="password")

        if st.button("Create Account"):
            if new_user in users:
                st.error("User already exists")
            else:
                users[new_user] = hash_password(new_pass)
                save_users(users)
                st.success("Account created successfully!")

    else:
        username = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if username in users and users[username] == hash_password(password):
                st.session_state.user = username
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid credentials")

# ==============================
# MAIN DASHBOARD
# ==============================
else:

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "History"])

    st.sidebar.write(f"Logged in as: {st.session_state.user}")

    if st.sidebar.button("Logout"):
        st.session_state.user = None
        st.rerun()

    if page == "Dashboard":

        st.title("🌿 Eco-Insight Dashboard")

        location = st.text_input("Enter Location Name")
        years_input = st.text_input("Enter Years of Uploaded Images (comma separated)")
        future_year = st.number_input("Enter Future Year to Predict", 2025, 2100, 2035)

        uploaded_files = st.file_uploader(
            "Upload 2-3 Satellite Images of Same Location",
            type=["jpg", "png"],
            accept_multiple_files=True
        )

        if st.button("Analyze"):

            if uploaded_files and years_input:

                years = list(map(int, years_input.split(",")))
                ndvi_values = []

                for file in uploaded_files:
                    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                    image = cv2.imdecode(file_bytes, 1)
                    ndvi = calculate_ndvi(image)
                    ndvi_values.append(ndvi)

                # Linear Regression
                X = np.array(years).reshape(-1, 1)
                y = np.array(ndvi_values)

                model = LinearRegression()
                model.fit(X, y)

                predicted_future = model.predict([[future_year]])[0]

                # Trend
                status = "Wetland Stable"
                if predicted_future < min(ndvi_values):
                    status = "⚠ Wetland Destruction Risk"

                # Display Results
                st.subheader("Current NDVI Values")
                for y_val, nd in zip(years, ndvi_values):
                    st.write(f"{y_val}: {round(nd, 3)}")

                st.subheader(f"Predicted NDVI for {future_year}")
                st.success(round(predicted_future, 3))

                st.subheader("Wetland Status")
                st.info(status)

                # Plot
                plt.figure()
                plt.plot(years, ndvi_values, marker='o')
                plt.scatter(future_year, predicted_future)
                plt.xlabel("Year")
                plt.ylabel("NDVI")
                st.pyplot(plt)

                # Save History
                history = load_history()
                new_row = pd.DataFrame([{
                    "username": st.session_state.user,
                    "location": location,
                    "year": years[-1],
                    "ndvi": ndvi_values[-1],
                    "future_year": future_year,
                    "future_ndvi": predicted_future,
                    "status": status,
                    "timestamp": datetime.now()
                }])

                history = pd.concat([history, new_row], ignore_index=True)
                save_history(history)

                # Download Report
                report = f"""
                Location: {location}
                Years: {years}
                NDVI Values: {ndvi_values}
                Future Year: {future_year}
                Predicted NDVI: {predicted_future}
                Status: {status}
                """

                st.download_button(
                    "Download Report",
                    report,
                    file_name="wetland_report.txt"
                )

            else:
                st.warning("Please upload images and enter years")

    else:
        st.title("📜 User History")

        history = load_history()
        user_history = history[history["username"] == st.session_state.user]

        if not user_history.empty:
            st.dataframe(user_history)
        else:
            st.info("No history found.")
