# -
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
import numpy as np
from colorama import Fore
plt.style.use('dark_background')
# -
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
# from colorama import Fore
# from sklearn.metrics import mean_squared_error
# -
import shaka_model
# -
from shaka_config import conf
from shaka_criterion import criterion
# -



config = conf()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

shaka_data = pd.read_csv(r'D:\Pycharm\API_Ex\shaka_data.csv', index_col = 0)

df = shaka_data['view_count']
# print(df)
# visualize
# plt.figure(figsize = (16, 8))
# plt.title('view_count')
# plt.plot(df)
# plt.xlabel('count')
# plt.ylabel('number of views')
# plt.show()
# to numpy array
dataset = df.values
# Get the number of rows to train the model on
training_data_len = int(len(dataset) * 0.5)
# print(training_data_len)
# 68
# https://www.youtube.com/watch?v=8A6TEjG2DNw&t=3230s
train_data = dataset[:training_data_len]
test_data = dataset[training_data_len:]

scaler = MinMaxScaler()
scaler = scaler.fit(np.expand_dims(train_data, axis = 1))

train_data = scaler.transform(np.expand_dims(train_data, axis = 1))
test_data = scaler.transform(np.expand_dims(test_data, axis = 1))

def sliding_windows(data, seq_length):
    xs = []
    ys = []

    for i in range(len(data) - seq_length - 1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)

seq_length = config['sequence_length']
X_train, y_train = sliding_windows(train_data, seq_length)
X_test, y_test = sliding_windows(test_data, seq_length)

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()

X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()
# print(X_train.shape, X_test.shape)
# print(y_train.shape)


def train_model(model, train_data, train_labels, test_data = None, test_labels = None):
    optimizer = optim.Adam(model.parameters(), lr = config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           patience=80,
                                                           factor=0.9,
                                                           min_lr=1e-7,
                                                           eps=1e-08)

    num_epochs = config['num_epoch']

    train_hist = np.zeros(num_epochs)
    test_hist = np.zeros(num_epochs)

    for t in range(num_epochs):
        model.train()


        pred = model(X_train.to(device))
        optimizer.zero_grad()

        loss = criterion(pred, y_train.to(device), f'Epoch-{t}-Train Loss : ')

        if test_data is not None:
            model.eval()
            with torch.no_grad():
                y_test_pred = model(X_test.to(device))
                test_loss = criterion(y_test_pred, y_test.to(device), 'Test Loss : ')
                scheduler.step(test_loss)
            test_hist[t] = test_loss.item()


        train_hist[t] = loss.item()
        loss.backward()
        optimizer.step()

    # del optimizer
    return model.eval(), train_hist, test_hist

model = shaka_model.LSTM(config['input_size'],
                         config['hidden_size'],
                         config['num_layers'],
                         config['num_classes'],
                         device = device).to(device)
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

model.apply(init_weights)

model, train_hist, test_hist = train_model(model, X_train, y_train, X_test, y_test)

# plt.figure(figsize=(6,4))
# plt.plot(train_hist, 'r+')
# plt.plot(test_hist, 'b-')
# plt.legend()
# plt.savefig('LSTM_train_and_valid_pred.png')
# plt.show()
new_length = 100
with torch.no_grad():
    test_seq = train_data[-seq_length:]
    test_seq = torch.from_numpy(test_seq.reshape(1, seq_length, 1)).float()
    # [1, 10, 1] -> [1,9,1] + [1,1,1] -> [1,10,1]としたい
    # print('nya--n!!', test_seq.shape)
    preds = []
    # print(Fore.RED,test_seq.shape)
    for _ in range(len(X_test)):
        y_test_pred = model(test_seq.to(device))

        pred = y_test_pred.view(-1).clone().detach().item()
        # print('PRED !! ', pred)
        # print('type:pred:', type(pred), pred.dtype)
        preds.append(pred)

        new_seq = test_seq.reshape(seq_length)
        # new_seq = np.append(new_seq, [pred])
        # print("1 new_seq's shape", new_seq.shape)
        # print(new_seq[1:].shape, torch.tensor(pred).shape)
        new_seq = torch.cat((new_seq[1:], torch.tensor(pred).reshape(1)))
        # print("2 new_seq's shape", new_seq.shape)
        test_seq = new_seq.reshape(1, seq_length, 1)
        # test_seq = torch.as_tensor(new_seq).view(1, seq_length, 1).float()
    new = []

    for i in range(new_length):
        # print(test_seq.flatten())
        y_test_pred = model(test_seq.to(device))

        pred = y_test_pred.view(-1).clone().detach().item()
        # print('PRED !! ', pred)
        # print('type:pred:', type(pred), pred.dtype)
        # print(pred)
        new.append(pred)

        new_seq = test_seq.reshape(seq_length)
        # new_seq = np.append(new_seq, [pred])
        # print("1 new_seq's shape", new_seq.shape)
        # print(new_seq[1:].shape, torch.tensor(pred).shape)
        new_seq = torch.cat((new_seq[1:], torch.tensor(pred).reshape(1)))
        # print("2 new_seq's shape", new_seq.shape)
        test_seq = new_seq.reshape(1, seq_length, 1)
        # test_seq = torch.as_tensor(new_seq).view(1, seq_length, 1).float()


# print('y_test:', y_test.shape)
# print('y_test_pred:', len(preds))

true_cases = scaler.inverse_transform(
    np.expand_dims(y_test.flatten().numpy(), axis = 0)
).flatten()

predicted_cases = scaler.inverse_transform(
    np.expand_dims(preds, axis = 0)
).flatten()

# これを忘れるな
new_cases = scaler.inverse_transform(
    np.expand_dims(new, axis = 0)
).flatten()

del model
torch.cuda.empty_cache()

plt.plot(df.index[:training_data_len+seq_length+1],
        dataset[:training_data_len+seq_length+1]
        , 'y--', label = 'train_data')
plt.plot(df.index[training_data_len+seq_length+1:],
         true_cases,'g--' , label = 'test_data')
plt.plot(df.index[training_data_len+seq_length+1:],
         predicted_cases,'r--', label = 'predict_data')
plt.plot(list(range(df.shape[0]-1,df.shape[0]-1+new_length)),
         new_cases, 'b', label = 'new_predict_data')
plt.legend()
plt.savefig('20210223lstm.png')
plt.show()
