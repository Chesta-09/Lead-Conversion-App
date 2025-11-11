import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go

# ------------------------- #
# Streamlit Configuration
# ------------------------- #
st.set_page_config(page_title="Lead Conversion Predictor", page_icon="🤖", layout="centered")
st.title("🤖 Lead Conversion Prediction App")
st.markdown("""
Predict whether a lead will **convert** using Logistic Regression.  
Upload your own dataset or use the default **Leads.csv** to train the model.
""")

# ------------------------- #
# Preprocessing + Model Training
# ------------------------- #
def preprocess_and_train(df):
    encoder = LabelEncoder()
    df = df.copy()

    # Drop columns with unique values (like IDs)
    for col in df.columns:
        if df[col].nunique() == len(df):
            df.drop(col, axis=1, inplace=True)

    # Encode categorical features
    for col in df.columns:
        if df[col].dtype == 'object':
            if df[col].nunique() <= 10:
                df[col + "_encoded"] = encoder.fit_transform(df[col].astype(str))
            else:
                # Frequency encoding for high-cardinality columns
                df[col + "_encoded"] = df[col].map(df[col].value_counts(normalize=True))

    # Drop original categorical columns
    df.drop(df.select_dtypes(include="object").columns, axis=1, inplace=True)
    df.fillna(df.mean(), inplace=True)

    # Check for target column
    if "Converted" not in df.columns:
        raise ValueError("Target column 'Converted' not found in the dataset!")

    # Split features/target
    X = df.drop("Converted", axis=1)
    y = df["Converted"]

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    # Train model
    model = LogisticRegression(max_iter=100000)
    model.fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    return model, X, accuracy

# ------------------------- #
# File Upload + Model Train
# ------------------------- #
uploaded_file = st.file_uploader("📤 Upload a CSV file (must contain a 'Converted' column)", type=["csv"])

if uploaded_file is not None:
    try:
        # Make sure the file pointer is at the start (UploadedFile may have been read)
        try:
            uploaded_file.seek(0)
        except Exception:
            # Some file-like objects may not support seek; ignore if not supported
            pass

        # Read uploaded CSV and handle empty-data specifically
        try:
            df = pd.read_csv(uploaded_file)
        except pd.errors.EmptyDataError:
            st.error("❌ Uploaded CSV contains no data or no columns. Please check the file and try again.")
            st.stop()

        if df.empty:
            st.error("❌ Uploaded CSV is empty. Please upload a valid CSV file.")
            st.stop()

        model, X, accuracy = preprocess_and_train(df)
        st.success(f"✅ Model retrained successfully with uploaded data. Accuracy: **{accuracy*100:.2f}%**")
    except Exception as e:
        st.error(f"❌ Error while processing file: {e}")
        st.stop()
else:
    try:
        try:
            df = pd.read_csv("Leads.csv")
        except pd.errors.EmptyDataError:
            st.error("⚠️ Default Leads.csv is empty or invalid. Please provide a valid CSV or upload one.")
            st.stop()

        if df.empty:
            st.error("⚠️ Default Leads.csv is empty. Please provide a dataset or upload a CSV.")
            st.stop()

        model, X, accuracy = preprocess_and_train(df)
        st.info("📘 Using default Leads.csv dataset")
        st.success(f"✅ Default model trained. Accuracy: **{accuracy*100:.2f}%**")
    except FileNotFoundError:
        st.error("⚠️ No file uploaded and default Leads.csv not found. Please upload a CSV file to continue.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error while loading default dataset: {e}")
        st.stop()

# ------------------------- #
# User Input Section
# ------------------------- #
st.sidebar.header("🧩 Enter Lead Features")

# Re-use the already-loaded dataframe (avoids re-reading an exhausted uploaded file buffer)
original_df = df.copy()

# Separate numeric and categorical columns (excluding target)
numeric_cols = original_df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = original_df.select_dtypes(exclude=np.number).columns.tolist()

if "Converted" in numeric_cols:
    numeric_cols.remove("Converted")
if "Converted" in categorical_cols:
    categorical_cols.remove("Converted")

user_input = {}

# Dropdowns for categorical features
for col in categorical_cols:
    unique_vals = sorted(original_df[col].dropna().unique().tolist())
    if len(unique_vals) > 0:
        user_input[col] = st.sidebar.selectbox(f"{col}", options=unique_vals)
    else:
        user_input[col] = None

# Sliders for numeric features
for col in numeric_cols:
    mean_val = float(original_df[col].mean())
    min_val = float(original_df[col].min())
    max_val = float(original_df[col].max())

    if min_val == max_val:
        user_input[col] = min_val
        st.sidebar.info(f"{col} has a constant value of {min_val}")
    else:
        user_input[col] = st.sidebar.slider(f"{col}", min_value=min_val, max_value=max_val, value=mean_val)

# Convert to DataFrame for prediction
input_df = pd.DataFrame([user_input])

# ------------------------- #
# Encode user input same way as training
# ------------------------- #
encoded_input = input_df.copy()

for col in encoded_input.columns:
    if encoded_input[col].dtype == 'object':
        if col in original_df.columns:
            if original_df[col].nunique() <= 10:
                le = LabelEncoder()
                le.fit(original_df[col].astype(str))
                encoded_input[col] = le.transform(encoded_input[col].astype(str))
            else:
                freq_map = original_df[col].value_counts(normalize=True).to_dict()
                encoded_input[col] = encoded_input[col].map(freq_map).fillna(0)

# Match columns to trained X
encoded_input = encoded_input.reindex(columns=X.columns, fill_value=0)

# ------------------------- #
# Prediction Button
# ------------------------- #
if st.button("🔍 Predict Lead Conversion"):
    prediction = model.predict(encoded_input)[0]
    probability = model.predict_proba(encoded_input)[0][1]
    st.subheader("📊 Prediction Results")
    st.write(f"**Converted:** {'✅ Yes' if prediction == 1 else '❌ No'}")
    st.write(f"**Conversion Probability:** `{probability*100:.2f}%`")

    # Gauge chart for visual effect
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': "Conversion Probability (%)"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "green" if probability > 0.5 else "red"},
               'steps': [{'range': [0, 50], 'color': "#ffcccc"},
                         {'range': [50, 100], 'color': "#ccffcc"}]}))
    st.plotly_chart(fig, use_container_width=True)

# ------------------------- #
# Data Preview
# ------------------------- #
with st.expander("📂 Preview Training Data"):
    st.dataframe(X.head())

st.caption("Built with ❤️ using Streamlit and Scikit-learn.")
