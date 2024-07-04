import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=configs.enc_in, out_channels=64, kernel_size=3)
        self.drop1 = nn.Dropout(configs.dropout)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=configs.enc_in, out_channels=64, kernel_size=5)
        self.drop2 = nn.Dropout(configs.dropout)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(in_channels=configs.enc_in, out_channels=64, kernel_size=11)
        self.drop3 = nn.Dropout(configs.dropout)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        flat_dim = self._get_flat_dim((configs.seq_len, configs.enc_in))
        self.fc1 = nn.Linear(flat_dim, 100)
        self.fc2 = nn.Linear(100, configs.num_class)

    def _get_flat_dim(self, input_shape):
        x = torch.randn(1, *input_shape).permute(0, 2, 1)  # Permute to [batch_size, in_channels, seq_len]

        x1 = self.pool1(self.drop1(self.conv1(x)))
        x2 = self.pool2(self.drop2(self.conv2(x)))
        x3 = self.pool3(self.drop3(self.conv3(x)))

        x1_flat = x1.reshape(1, -1).size(1)
        x2_flat = x2.reshape(1, -1).size(1)
        x3_flat = x3.reshape(1, -1).size(1)

        return x1_flat + x2_flat + x3_flat

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        x = x.permute(0, 2, 1)  # Permute to [batch_size, in_channels, seq_len]

        x1 = self.pool1(self.drop1(self.conv1(x)))
        x1 = x1.reshape(x1.size(0), -1)

        x2 = self.pool2(self.drop2(self.conv2(x)))
        x2 = x2.reshape(x2.size(0), -1)

        x3 = self.pool3(self.drop3(self.conv3(x)))
        x3 = x3.reshape(x3.size(0), -1)

        x = torch.cat((x1, x2, x3), dim=1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path + '/checkpoint.pth'))
