import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cpu")
columns = ['GZ3', 'bk', 'NKTR', 'NKTD', 'GZ1', 'DGK', 'ALPS']


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(len(columns), 10, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(10, 10, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(10, 1, kernel_size=5, padding=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.sigmoid(self.conv3(x))
        return x


def predict_by_df(df, model):
  df = df.sort_values('depth, m')
  x = df[columns].values
  x = torch.tensor(x, dtype=torch.float, device=device)
  x = x.unsqueeze(0)
  x = x.permute(0, 2, 1)
  with torch.no_grad():
    model.eval()
    y_pred = model(x).cpu().numpy().squeeze()
  return y_pred