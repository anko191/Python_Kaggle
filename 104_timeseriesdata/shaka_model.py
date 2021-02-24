import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class LSTM(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 num_classes,
                 device = 'cpu'):
        super(LSTM, self).__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers,
                            batch_first=True,
                            dropout = 0.2)
        # N x time_seq x features
        self.fc = nn.Linear(hidden_size, num_classes)
        # self.dropout = F.dropout(0.2)

    def forward(self, x):
        h0 = Variable(torch.zeros(self.num_layers, x.size(0),
                         self.hidden_size).to(self.device))
        # 短期記憶用
        c0 = Variable(torch.zeros(self.num_layers, x.size(0),
                         self.hidden_size).to(self.device))
        # 長期記憶用

        # forward prop
        # print(self.device)
        _, (h_n, c_n) = self.lstm(x, (h0, c0))
        # 出力なんだこれ？？
        # print(out.shape)
        # out = h_n.view(-1, self.hidden_size)

        final_state = h_n.view(self.num_layers, x.size(0), self.hidden_size)[-1]

        # out, _ = self.lstm(out, (h1, c1))
        out = F.relu(self.fc(final_state))

        # print('x:',x.shape)
        # print('out:',out.shape)
        # nn.Linear に対応するように nxm \times mxp
        # batch_size, -1, hidden_size
        return out
