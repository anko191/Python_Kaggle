from colorama import Fore
# from sklearn.metrics import mean_squared_error
import torch.nn as nn
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def criterion(pred, real, name = 'Base:'):
    """
    this criterion shows RMSE

    :param pred: predictions value
    :param real: real value ex.testdata
    :param name: optional
    :return: None
    """
    # rmse = mean_squared_error(pred, real, squared=True)
    criterion = nn.MSELoss().to(device)
    loss = criterion(pred.float(), real.float())
    print(Fore.CYAN + f'{name} {loss} !!')
    return loss