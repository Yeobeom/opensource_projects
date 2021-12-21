import binance_data_load as BDL
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error
from datetime import datetime
import time
import numpy as np


api_key = ""
secret = ""
t_frame = '1d' # 1m, 15m, 30m, 1h, 12h, 1d
symbol = 'ETH/USDT'

EPOCH_MAX = 30
EPOCH_LOG = 1
OPTIMIZER_PARAM = {'lr': 0.01}
SCHEDULER_PARAM = {'step_size': 5, 'gamma': 0.5}
USE_CUDA = torch.cuda.is_available()
SAVE_MODEL = f'./data/ETH_LSTM_{t_frame}.pt' # Make empty('') if you don't want save the model
RANDOM_SEED = 777
DATA_LOADER_PARAM = {'batch_size': 50, 'shuffle': False}

class ETH_LSTM(nn.Module):
    def __init__(self, input_size, output_size):
        super(ETH_LSTM, self).__init__()
        self.LSTM1 = torch.nn.LSTM(input_size, 128)
        self.LSTM2 = torch.nn.LSTM(128,128)
        self.fc = torch.nn.Linear(128, output_size)

    def forward(self, x):
        output, hidden = self.LSTM1(x)
        output, hidden = self.LSTM2(output)
        x = self.fc(output) # Use output of the last sequence
        return x

def train(model, batch_data, loss_func, optimizer):
    model.train()  # Notify layers (e.g. DropOut, BatchNorm) that it’s now training
    train_loss, n_data = 0, 0
    dev = next(model.parameters()).device
    for batch_idx, (x, y) in enumerate(batch_data):
        x, y = x.to(dev), y.to(dev)
        optimizer.zero_grad()
        x = x.unsqueeze(0)
        x = x.unsqueeze(1)
        output = model(x)
        y = y.unsqueeze(0)
        y = y.unsqueeze(1)
        #print(output.shape,y.shape)
        loss = loss_func(output, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        n_data += len(y)
    return train_loss / n_data

def predict_price(data, model,data_normalizer,target_normalizer):
    model.eval()
    with torch.no_grad():
        dev = next(model.parameters()).device
        data = data.reshape(1,-1)
        data = data_normalizer.transform(data)
        data = torch.tensor(data, dtype=torch.float32, device=dev)
        data = data.unsqueeze(0)
        output = model(data)
        output = output.reshape(1,-1)
        output = output.to('cpu').numpy()
        output = target_normalizer.inverse_transform(output)
        output = output.reshape(-1)
        output = output.tolist()
        return output


if __name__ == '__main__':
    # 0. Preparation
    torch.manual_seed(RANDOM_SEED)
    if USE_CUDA:
        torch.cuda.manual_seed_all(RANDOM_SEED)
    dev = torch.device('cuda' if USE_CUDA else 'cpu')

    # 1. Load the coinprice dataset
    bdl = BDL.binance_data(api_key,secret,symbol,t_frame)
    if api_key:
        bdl.updating_coin_csv()
    coindata = bdl.Load_Coin_Data()
    coin_data = coindata.data
    coin_targets = coindata.targets

    

    # 1.1 normalizeData
    data_normalizer = preprocessing.MinMaxScaler()
    target_normalizer = preprocessing.MinMaxScaler()
    coin_data_normalized = data_normalizer.fit_transform(coin_data)
    coin_targets_normalized = target_normalizer.fit_transform(coin_targets)
    #test_data_normalized = data_normalizer.fit_transform(test_data)
    
    # 1.2 fitting data to Tensor
    x = torch.tensor(coin_data_normalized, dtype=torch.float32, device=dev)
    y = torch.tensor(coin_targets_normalized, dtype=torch.float32, device=dev)
    data_train = torch.utils.data.TensorDataset(x,y)
    loader_train = torch.utils.data.DataLoader(data_train, **DATA_LOADER_PARAM)

    # 2. Instantiate a model, loss function, and optimizer
    model = ETH_LSTM(x.shape[1],y.shape[1]).to(dev)
    loss_func = nn.MSELoss() # for regression
    optimizer = torch.optim.Adam(model.parameters(), **OPTIMIZER_PARAM)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **SCHEDULER_PARAM)

    # 3.1. Train the model
    loss_list = []
    start = time.time()
    for epoch in range(1, EPOCH_MAX + 1):
        train_loss = train(model, data_train, loss_func, optimizer)
        scheduler.step()

        loss_list.append([epoch, train_loss])
        if epoch % EPOCH_LOG == 0:
            elapse = (time.time() - start) / 60
            print(f'{epoch:>6} ({elapse:>6.2f} min), TrLoss={train_loss:.6f}, lr={scheduler.get_last_lr()}')
    elapse = (time.time() - start) / 60

    # 3.2. Save the trained model if necessary
    if SAVE_MODEL:
        torch.save(model.state_dict(), SAVE_MODEL)

    # 3.3 predict_data
    predicts = [predict_price(row,model,data_normalizer,target_normalizer) for row in coin_data]

    # 4.1. Visualize the loss curves
    plt.title(f'Training Loss (time: {elapse:.2f} [min] @ CUDA: {USE_CUDA})')
    loss_array = np.array(loss_list)
    plt.plot(loss_array[:, 0], loss_array[:, 1], label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss values')
    plt.xlim(loss_array[0, 0], loss_array[-1, 0])
    plt.grid()
    plt.legend()
    plt.show()

    # 4.2. Visualize the curves
    plt.figure()
    plt.title('Real_data vs predicted_data')
    plt.plot([datetime.fromtimestamp(float(row[0])/1000) for row in coindata.features],[row[3] for row in coindata.data], label='Real_price')
    plt.plot([datetime.fromtimestamp(float(row[0])/1000) for row in coindata.features],[row[1] for row in predicts], label='predicted_price')
    plt.xlabel('DATE')
    plt.ylabel('price')
    plt.grid()
    plt.legend()
    plt.show()
    
    # 4.3 error
    mae_value = mean_absolute_error(coin_targets,predicts)
    print(f'MAE : {mae_value}')
    
    # 5. predict price
    recent_data = np.array(coindata.recent_data[1:6])
    predicted_price = predict_price(recent_data,model,data_normalizer,target_normalizer)
    addtime = {'1m':60000, '15m':900000, '30m':1800000,'1h':3600000, '12h':43200000,'1d':86400000}
    print(f'{datetime.fromtimestamp(float(coindata.recent_data[6]+addtime[t_frame])/1000)} 기준:')
    print(f'예상 시작가: {predicted_price[0]}, 예상 종가: {predicted_price[1]}')