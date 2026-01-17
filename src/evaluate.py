import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
from data import StockDataLoader

# Configuration
DATA_PATH = 'dataset/TSLA.csv'
SEQ_LENGTH = 60
HORIZONS = [1, 5, 10]
MODELS_DIR = 'models'
RESULTS_DIR = 'results'

def evaluate_models():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    loader = StockDataLoader(DATA_PATH, sequence_length=SEQ_LENGTH)
    # Preload data to get scaler
    loader.preprocess()
    scaler = loader.scaler
    
    results_summary = []

    for horizon in HORIZONS:
        print(f"\nEvaluating Horizon: {horizon} Day(s)")
        X_train, y_train, X_test, y_test = loader.get_train_test_split(prediction_days=horizon)
        
        # Prepare actual prices for plotting (inverse transform)
        # y_test is scaled.
        actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
        
        plt.figure(figsize=(14, 6))
        plt.plot(actual_prices, color='black', label='Actual Price')
        
        for model_type in ['SimpleRNN', 'LSTM']:
            model_path = os.path.join(MODELS_DIR, f"{model_type}_{horizon}day.h5")
            if not os.path.exists(model_path):
                print(f"Model not found: {model_path}")
                continue
                
            model = load_model(model_path)
            predictions = model.predict(X_test)
            predicted_prices = scaler.inverse_transform(predictions)
            
            mse = mean_squared_error(actual_prices, predicted_prices)
            results_summary.append({
                'Horizon': horizon,
                'Model': model_type,
                'MSE': mse
            })
            
            plt.plot(predicted_prices, label=f'{model_type} (MSE: {mse:.2f})')
        
        plt.title(f'Tesla Stock Price Prediction - {horizon} Day(s) Horizon')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.savefig(os.path.join(RESULTS_DIR, f'prediction_{horizon}day.png'))
        plt.close()
        print(f"Saved plot to results/prediction_{horizon}day.png")

    # Save metrics
    results_df = pd.DataFrame(results_summary)
    results_df.to_csv(os.path.join(RESULTS_DIR, 'metrics.csv'), index=False)
    print("\nEvaluation Complete. Metrics saved to results/metrics.csv")
    print(results_df)

if __name__ == "__main__":
    evaluate_models()
