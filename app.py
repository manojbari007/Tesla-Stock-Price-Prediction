import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from src.data import StockDataLoader

# Configuration
st.set_page_config(page_title="Tesla Stock Prediction", layout="wide")
DATA_PATH = 'dataset/TSLA.csv'
MODELS_DIR = 'models'
SEQ_LENGTH = 60

@st.cache_data
def load_data():
    loader = StockDataLoader(DATA_PATH, sequence_length=SEQ_LENGTH)
    df = loader.load_data()
    return df, loader

def load_trained_model(model_type, horizon):
    model_path = os.path.join(MODELS_DIR, f"{model_type}_{horizon}day.h5")
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

def main():
    st.title("Tesla Stock Price Prediction 📈")
    st.markdown("""
    This application attempts to predict Tesla stock prices using Deep Learning models (**SimpleRNN** and **LSTM**).
    """)

    # Sidebar
    st.sidebar.header("Configuration")
    model_type = st.sidebar.selectbox("Select Model", ["SimpleRNN", "LSTM"])
    horizon = st.sidebar.selectbox("Prediction Horizon (Days)", [1, 5, 10])

    # Load Data
    try:
        df, loader = load_data()
        st.subheader("Historical Stock Data")
        st.line_chart(df['Adj Close'])
        
        # Show raw data
        with st.expander("See Raw Data"):
            st.dataframe(df.tail(100))

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    # Prediction
    st.subheader(f"Prediction Analysis ({model_type} - {horizon} Day Horizon)")
    
    with st.spinner("Loading Model and Predicting..."):
        model = load_trained_model(model_type, horizon)
        
        if model:
            # Prepare data
            # We need to preprocess/scale again
            loader.preprocess() 
            X, y = loader.create_sequences(prediction_days=horizon)
            
            # Predict
            predictions = model.predict(X)
            
            # Inverse transform
            scaler = loader.scaler
            predicted_prices = scaler.inverse_transform(predictions)
            actual_prices = scaler.inverse_transform(y.reshape(-1, 1))
            
            # Create a localized dataframe for better plotting in Streamlit
            # We align the last N dates
            # y corresponds to date at index i + seq_len + horizon - 1
            # But simpler correlation: just plot them together.
            
            result_df = pd.DataFrame({
                "Actual": actual_prices.flatten(),
                "Predicted": predicted_prices.flatten()
            })
            
            st.line_chart(result_df)
            
            # Metrics
            mse = np.mean((predicted_prices - actual_prices) ** 2)
            st.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
            
            st.success("Prediction Complete!")
            
        else:
            st.warning(f"Model ({model_type}, {horizon} days) not found! Please run the training script first.")
            st.info(f"Expected model path: `models/{model_type}_{horizon}day.h5`")

if __name__ == "__main__":
    main()
