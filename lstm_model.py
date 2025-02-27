from torch import nn, optim, no_grad, cat
from data_preprocessing import training_set, val_set, test_set, y_scaler, dates
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from prediction_quality_metrics import *


args_dict = {
    "input_dim": 3,
    "hidden_dim": 50,
    "layer_dim": 2,
    "output_dim": 1,
    "batch_size": 24,
    "lr": 0.01,
    "num_epoch": 100,
    "model": "LSTM"
}


class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True) if model == "GRU" else nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, (h_n, c_n) = self.model(x)
        last_out = out[:, -1, :]
        out = self.fc(last_out)
        return out


def train_lstm(dataloader, model: Model, criterion, optimizer, is_training=False):
    total_loss = 0

    if is_training:
        model.train()
    else:
        model.eval()

    for index, (X, y) in enumerate(dataloader):
        outputs = model(X)
        loss = criterion(outputs.contiguous(), y.contiguous())
        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
    return total_loss


def evaluate_test_set(dataloader, model: Model):
    pred = []
    model.eval()
    with no_grad():
        for index, (X, y) in enumerate(dataloader):
            outputs = model(X)
            pred.append(outputs)
    return pred


train_dl = DataLoader(training_set, batch_size=args_dict['batch_size'], shuffle=True)
val_dl = DataLoader(val_set, batch_size=args_dict['batch_size'], shuffle=False)
test_dl = DataLoader(test_set, batch_size=args_dict['batch_size'], shuffle=False)
model = Model(input_dim=args_dict['input_dim'], hidden_dim=args_dict['hidden_dim'], layer_dim=args_dict['layer_dim'], output_dim=args_dict['output_dim'], model=args_dict['model'])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args_dict['lr'])


valid_losses, train_losses = [], []
for epoch in range(args_dict['num_epoch']):
    loss_train = train_lstm(train_dl, model=model, criterion=criterion, optimizer=optimizer, is_training=True)
    loss_val = train_lstm(val_dl, model=model, criterion=criterion, optimizer=optimizer)
    print(f'Epoch [{epoch+1}/{args_dict["num_epoch"]}] | loss train: {loss_train:.6f}, val: {loss_val:.6f}')
    valid_losses.append(loss_val)
    train_losses.append(loss_train)
pred_lst = evaluate_test_set(test_dl, model=model)

# Compute metrics for test set
inverse_output = y_scaler.inverse_transform(cat(pred_lst))
inverse_actual = y_scaler.inverse_transform(test_set.y)


mape_test = mean_absolute_percentage_error(inverse_actual, inverse_output)
rmse_test = root_mean_squared_error(inverse_actual, inverse_output)
mae_test = mean_absolute_error(inverse_actual, inverse_output)

print(f"Test Metrics: MAPE: {mape_test:.2f}%, RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}")


plt.figure(1)
plt.title("Training vs. Validation Losses")
plt.plot(train_losses, label='train', linestyle='-', linewidth=3, marker='o', markersize=6, color='b')
plt.plot(valid_losses, label='val', linestyle='-', linewidth=3, marker='o', markersize=6, color='r')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()

plt.figure(2)
plt.title("Test Set Predictions vs. Actual Values")
plt.plot(dates[-len(test_set.y):], inverse_output, label='pred', linestyle='-', linewidth=1, marker='o', markersize=3)
plt.plot(dates[-len(test_set.y):], inverse_actual, label='actual', linestyle='-', linewidth=1, marker='o', markersize=3)
plt.legend()
plt.show()










