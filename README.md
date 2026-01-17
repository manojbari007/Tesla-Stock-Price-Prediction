# 🚗 Tesla Stock Price Prediction using Deep Learning

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://manojbari007-tesla-stock-price-prediction.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black.svg)](https://github.com/manojbari007/Tesla-Stock-Price-Prediction)

## 🌐 Live Demo

**👉 [Try the Streamlit App](https://manojbari007-tesla-stock-price-prediction.streamlit.app)**

Experience the Tesla Stock Price Prediction model in action! Upload data, select prediction horizons, and visualize forecasts.

---

## Project Overview

This project implements deep learning models (SimpleRNN and LSTM) to predict Tesla stock prices for 1-day, 5-day, and 10-day horizons.

## 🎯 Skills Demonstrated

- Basic Python
- Data Visualization
- Data Cleaning & EDA
- Deep Learning (SimpleRNN and LSTM)

## 📊 Domain

Financial Services

## 🗂️ Project Structure

```
Tesla Stock Price Prediction/
├── dataset/
│   └── TSLA.csv                    # Tesla historical stock data
├── models/
│   ├── SimpleRNN_1day.h5           # SimpleRNN model for 1-day prediction
│   ├── SimpleRNN_5day.h5           # SimpleRNN model for 5-day prediction
│   ├── SimpleRNN_10day.h5          # SimpleRNN model for 10-day prediction
│   ├── LSTM_1day.h5                # LSTM model for 1-day prediction
│   ├── LSTM_5day.h5                # LSTM model for 5-day prediction
│   └── LSTM_10day.h5               # LSTM model for 10-day prediction
├── results/
│   ├── metrics.csv                 # Model performance metrics
│   └── *.png                       # Visualization plots
├── src/
│   ├── data.py                     # Data loading and preprocessing
│   ├── models.py                   # Model architectures
│   ├── train.py                    # Training script
│   └── evaluate.py                 # Evaluation script
├── Tesla_Stock_Prediction_DL.ipynb # Main Jupyter Notebook
├── app.py                          # Streamlit application
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🔧 Installation

1. Clone or download the project
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Option 1: Jupyter Notebook (Recommended)

Run the complete analysis step-by-step:

```bash
jupyter notebook Tesla_Stock_Prediction_DL.ipynb
```

### Option 2: Training Script

Train all models from command line:

```bash
cd src
python train.py
```

### Option 3: Streamlit App

Launch the interactive web application:

```bash
streamlit run app.py
```

## 📈 Business Use Cases

### 1. Stock Market Trading & Investment Strategies

- **Automated Trading**: Use predictions for algorithmic trading
- **Risk Management**: Assess future price movements for portfolio optimization

### 2. Financial Forecasting

- **Long-Term Investment Planning**: Predict trends for ETFs and mutual funds
- **Macroeconomic Analysis**: Compare with economic indicators

### 3. Business & Corporate Use Cases

- **Company Valuation**: Forecast revenue and profit trends
- **Competitor Analysis**: Apply to other EV stocks (Rivian, NIO, Lucid)

## 🧠 Model Architecture

### SimpleRNN

```
Input → SimpleRNN(50) → Dropout(0.2) → SimpleRNN(50) → Dropout(0.2) → Dense(1)
```

### LSTM

```
Input → LSTM(50) → Dropout(0.2) → LSTM(50) → Dropout(0.2) → Dense(1)
```

## 📊 Key Features

1. **Data Preprocessing**
   - Forward fill for missing values (time-series appropriate)
   - MinMaxScaler normalization (0-1 range)
   - Sequential train-test split (80-20)

2. **Model Training**
   - GridSearchCV for hyperparameter tuning
   - Early stopping to prevent overfitting
   - Model checkpointing for best weights

3. **Evaluation Metrics**
   - Mean Squared Error (MSE)
   - Root Mean Squared Error (RMSE)
   - Mean Absolute Error (MAE)
   - Mean Absolute Percentage Error (MAPE)

## 📝 Approach

1. **Problem Understanding**: Predict Tesla stock closing prices
2. **Data Preprocessing**: Load, clean, scale, create sequences
3. **Model Development**: Build SimpleRNN and LSTM models
4. **Hyperparameter Tuning**: GridSearchCV for optimal parameters
5. **Training**: Train with early stopping and validation
6. **Evaluation**: Compare predictions with actual prices
7. **Insights**: Analyze results and draw conclusions

## ⚠️ Limitations

- Stock prices are influenced by external factors not captured in historical data
- Market volatility and sudden events cannot be predicted
- Model assumes historical patterns will continue

## 🔮 Future Improvements

- Add sentiment analysis from news/social media
- Include technical indicators (RSI, MACD, Bollinger Bands)
- Incorporate macroeconomic data
- Try Transformer/Attention architectures

## 📄 License

This project is for educational purposes.

## 👨‍💻 Author

Tesla Stock Price Prediction Project - Deep Learning Implementation
