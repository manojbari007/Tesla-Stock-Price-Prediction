import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

class StockDataLoader:
    def __init__(self, file_path, sequence_length=60, feature_col='Adj Close'):
        self.file_path = file_path
        self.sequence_length = sequence_length
        self.feature_col = feature_col
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = None
        self.scaled_data = None
        
    def load_data(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found at {self.file_path}")
        
        df = pd.read_csv(self.file_path)
        
        # Convert Date to datetime and sort
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.sort_values('Date', inplace=True)
            df.set_index('Date', inplace=True)
            
        self.data = df
        return df

    def preprocess(self):
        if self.data is None:
            self.load_data()
            
        # Select feature
        data = self.data[[self.feature_col]].values
        
        # Handle missing values (simple forward fill if any, though stock data is usually clean)
        # In array form, we check for nans
        if np.isnan(data).any():
            data = pd.DataFrame(data).fillna(method='ffill').values
            
        # Scale
        self.scaled_data = self.scaler.fit_transform(data)
        return self.scaled_data

    def create_sequences(self, prediction_days=1):
        """
        Creates sequences X and y.
        X: Sequence of length `sequence_length`
        y: Target value `prediction_days` after the sequence
        """
        if self.scaled_data is None:
            self.preprocess()
            
        X = []
        y = []
        
        # We need to ensure we have enough data for the prediction horizon
        # data[i : i+seq_len] -> data[i+seq_len + prediction_days - 1]
        
        for i in range(self.sequence_length, len(self.scaled_data) - prediction_days + 1):
            X.append(self.scaled_data[i-self.sequence_length : i, 0])
            y.append(self.scaled_data[i + prediction_days - 1, 0])
            
        X, y = np.array(X), np.array(y)
        
        # Reshape X to (samples, time steps, features) for RNN/LSTM
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        return X, y

    def get_train_test_split(self, prediction_days=1, test_size=0.2):
        X, y = self.create_sequences(prediction_days)
        
        train_size = int(len(X) * (1 - test_size))
        
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    # fast test
    loader = StockDataLoader('dataset/TSLA.csv', sequence_length=60)
    X, y = loader.create_sequences(prediction_days=1)
    print(f"Details for 1 day prediction:")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
