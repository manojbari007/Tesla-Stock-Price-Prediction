import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from data import StockDataLoader
from models import build_simple_rnn, build_lstm

# Configuration
DATA_PATH = 'dataset/TSLA.csv'
SEQ_LENGTH = 60
HORIZONS = [1, 5, 10]
MODEL_TYPES = ['SimpleRNN', 'LSTM']
MODELS_DIR = 'models'

# Hyperparameter Grid
PARAM_GRID = {
    'units': [50, 100],
    'dropout_rate': [0.2],
    'learning_rate': [0.001]
}

def train_and_save():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    loader = StockDataLoader(DATA_PATH, sequence_length=SEQ_LENGTH)
    
    for horizon in HORIZONS:
        print(f"\n{'='*30}")
        print(f"Training for Horizon: {horizon} Day(s)")
        print(f"{'='*30}")
        
        X_train, y_train, X_test, y_test = loader.get_train_test_split(prediction_days=horizon)
        input_shape = (X_train.shape[1], 1)
        
        for model_type in MODEL_TYPES:
            print(f"\n--- Model: {model_type} ---")
            
            best_mse = float('inf')
            best_model = None
            best_params = {}
            
            # Grid Search
            for units in PARAM_GRID['units']:
                for dropout in PARAM_GRID['dropout_rate']:
                    for lr in PARAM_GRID['learning_rate']:
                        print(f"Training with units={units}, dropout={dropout}, lr={lr}...")
                        
                        if model_type == 'SimpleRNN':
                            model = build_simple_rnn(input_shape, units, dropout, lr)
                        else:
                            model = build_lstm(input_shape, units, dropout, lr)
                            
                        # Early stopping to prevent overfitting
                        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
                        
                        history = model.fit(
                            X_train, y_train,
                            epochs=10, # Kept low for demonstration speed, increase for real performance
                            batch_size=32,
                            validation_split=0.1,
                            callbacks=[early_stop],
                            verbose=0
                        )
                        
                        val_loss = min(history.history['val_loss'])
                        print(f"  -> Validation MSE: {val_loss:.6f}")
                        
                        if val_loss < best_mse:
                            best_mse = val_loss
                            best_model = model
                            best_params = {'units': units, 'dropout': dropout, 'lr': lr}
            
            print(f"Best {model_type} Params: {best_params} with MSE: {best_mse:.6f}")
            
            # Save best model
            save_path = os.path.join(MODELS_DIR, f"{model_type}_{horizon}day.h5")
            best_model.save(save_path)
            print(f"Saved best {model_type} model to {save_path}")

if __name__ == "__main__":
    train_and_save()
