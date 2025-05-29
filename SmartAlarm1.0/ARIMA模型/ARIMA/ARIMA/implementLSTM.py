# path_to_script.py
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Data preprocessing
def load_and_preprocess_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.rename(columns={'data': 'deal_data', 'time': 'time_data'}, inplace=True)
    df.set_index(['time_data'], inplace=True)  # Set index
    return df

def prepare_data(df: pd.DataFrame, time_steps: int) -> tuple[np.ndarray, np.ndarray]:
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['deal_data']])

    X, y = [], []
    for i in range(len(scaled_data) - time_steps):
        X.append(scaled_data[i:i + time_steps])
        y.append(scaled_data[i + time_steps])

    return np.array(X), np.array(y), scaler

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take the output of the last time step
        return out

# Train the model and make predictions
# Adjust the training loop in the train_and_predict function
def train_and_predict(df: pd.DataFrame, time_steps: int = 30, forecast_num: int = 7) -> None:
    X, y, scaler = prepare_data(df, time_steps)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # Split into train and test sets
    train_size = int(len(X_tensor) * 0.8)
    X_train, y_train = X_tensor[:train_size], y_tensor[:train_size]
    X_test, y_test = X_tensor[train_size:], y_tensor[train_size:]

    # Create and train LSTM model
    model = LSTMModel(input_size=1, hidden_size=50, num_layers=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    for epoch in range(100):  # Number of epochs
        optimizer.zero_grad()
        outputs = model(X_train)  # No need to add an extra dimension here
        loss = criterion(outputs, y_train.unsqueeze(-1))  # Keep y_train 2D
        loss.backward()
        optimizer.step()

    # Make predictions
    model.eval()
    with torch.no_grad():
        predicted = model(X_test)
        predicted = scaler.inverse_transform(predicted.numpy())

    # Visualize results
    plt.figure(figsize=(12, 6))
    plt.plot(df.index[train_size + time_steps:], df['deal_data'].values[train_size + time_steps:], color='blue', label='Actual Data')
    plt.plot(df.index[train_size + time_steps:], predicted, color='red', label='Predicted Data')
    plt.title('Actual vs Predicted Data')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.legend()
    plt.show()

    # Evaluate model
    mse = mean_squared_error(df['deal_data'].values[train_size + time_steps:], predicted)
    print(f'Mean Squared Error: {mse}')


if __name__ == '__main__':
    path = 'D:\\SmartAlarm\\相关材料\\ARIMA模型\\ARIMA\\ARIMA\\时间序列模型测试数据.xlsx'
    df = load_and_preprocess_data(path)
    train_and_predict(df, time_steps=30, forecast_num=7)  # Model invocation
