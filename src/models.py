from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, Dropout
from tensorflow.keras.optimizers import Adam

def build_simple_rnn(input_shape, units=50, dropout_rate=0.2, learning_rate=0.001):
    model = Sequential()
    # First RNN layer
    model.add(SimpleRNN(units=units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    
    # Second RNN layer
    model.add(SimpleRNN(units=units, return_sequences=False))
    model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(Dense(units=1))
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

def build_lstm(input_shape, units=50, dropout_rate=0.2, learning_rate=0.001):
    model = Sequential()
    # First LSTM layer
    model.add(LSTM(units=units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    
    # Second LSTM layer
    model.add(LSTM(units=units, return_sequences=False))
    model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(Dense(units=1))
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model
