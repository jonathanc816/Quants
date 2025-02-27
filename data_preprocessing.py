import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset

ticker = yf.Ticker('KO')
features = ['Close', 'Volume', 'Open']

# Fetch historical data for the last 5 years
historical_data = ticker.history(period='5y')[features]

historical_data.reset_index(inplace=True)

# Extract dates for later use
dates = historical_data['Date']

# Drop the 'Date' column from the main DataFrame
historical_data.drop("Date", axis=1, inplace=True)


print(historical_data.head(5))
print(historical_data.tail(5))


class DataPreprocess:
    def __init__(self, data):
        self.data = data.values
        self.shape = data.shape

    def generate_sliding_windows(self, window_size: int, output_feature: int):
        X, y = [], []
        for i in range(self.shape[0] - window_size):
            X.append(self.data[i:i + window_size])
            y.append(self.data[i + window_size, output_feature])
        return X, y

    @staticmethod
    def split_data(x, y, need_val: bool):
        if not need_val:
            train_size = int(len(x) * 0.8)
            x_training_set, x_test_set = x[:train_size], x[train_size:]
            y_training_set, y_test_set = y[:train_size], y[train_size:]
            return np.array(x_training_set), np.array(x_test_set), np.array(y_training_set), np.array(y_test_set)
        else:
            train_size = int(len(x) * 0.6)
            val_size = int(len(x) * 0.2)
            x_training_set, x_val_set, x_test_set = x[:train_size], x[train_size:train_size+val_size], x[train_size+val_size:]
            y_training_set, y_val_set, y_test_set = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]
            return np.array(x_training_set),np.array(x_val_set), np.array(x_test_set), np.array(y_training_set), np.array(y_val_set), np.array(y_test_set)
    @staticmethod
    def scale_data(x_train, x_val, x_test, y_train, y_val, y_test):
        scaler = MinMaxScaler()
        x_train_scaled = scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
        x_val_scaled = scaler.transform(x_val.reshape(-1, x_val.shape[-1])).reshape(x_val.shape)
        x_test_scaled = scaler.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)

        y_scaler = MinMaxScaler()
        if len(y_train.shape) == 1:
            y_train = y_train.reshape(-1, 1)
            y_val = y_val.reshape(-1, 1)
            y_test = y_test.reshape(-1, 1)
        y_train_scaled = y_scaler.fit_transform(y_train)
        y_val_scaled = y_scaler.transform(y_val)
        y_test_scaled = y_scaler.transform(y_test)

        return x_train_scaled, x_val_scaled, x_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled, y_scaler


class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        sample = self.X[item]
        label = self.y[item]
        return sample, label


dp = DataPreprocess(historical_data)
X, y = dp.generate_sliding_windows(window_size=20, output_feature=0)
x_training_set, x_test_set, x_val_set, y_training_set, y_val_set, y_test_set = dp.split_data(X, y, need_val=True)
x_train_scaled, x_val_scaled, x_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled, y_scaler = dp.scale_data(x_training_set, x_test_set, x_val_set, y_training_set, y_val_set, y_test_set)

training_set = StockDataset(x_train_scaled, y_train_scaled)
val_set = StockDataset(x_val_scaled, y_val_scaled)
test_set = StockDataset(x_test_scaled, y_test_scaled)

print(f"{'Dataset':<25}{'Shape'}")
print(f"{'Original Data':<25}{dp.shape}")
print(f"{'x_train_scaled':<25}{x_train_scaled.shape}")
print(f"{'x_val_scaled':<25}{x_val_scaled.shape}")
print(f"{'x_test_scaled':<25}{x_test_scaled.shape}")
print(f"{'y_train_scaled':<25}{y_train_scaled.shape}")
print(f"{'y_val_scaled':<25}{y_val_scaled.shape}")
print(f"{'y_test_scaled':<25}{y_test_scaled.shape}")










