
import torch

import torch.nn as nn
import torch.nn.functional as F

from collections import deque
from utils.custom_modules import Flatten, TemporalBatchNorm

class ContextConditioning(nn.Module):
    def __init__(self, zdim=128, n_lstm_layers=1, activation=nn.ELU, normalize_input=False,
                 spec_out=32, hidden_size=64, groupnorm=False):
        super(ContextConditioning, self).__init__()

        self.inplace = False

        if isinstance(activation, str):
            # if activation is provided as a string reference create a callable instance (if possible)
            activation = eval(activation)

        modules = []
        if normalize_input:
            print('Using input normalization!')
            modules.append(TemporalBatchNorm(78, affine=False))

        modules.extend([
            nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.GroupNorm(1, 24) if groupnorm else nn.BatchNorm2d(24),
            activation(self.inplace),
            nn.Conv2d(24, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.GroupNorm(1, 24) if groupnorm else nn.BatchNorm2d(24),
            activation(self.inplace),
            nn.MaxPool2d(2),

            nn.Conv2d(24, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.GroupNorm(1, 48) if groupnorm else nn.BatchNorm2d(48),
            activation(self.inplace),
            nn.Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.GroupNorm(1, 48) if groupnorm else nn.BatchNorm2d(48),
            activation(self.inplace),
            nn.MaxPool2d(2),

            nn.Conv2d(48, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.GroupNorm(1, 96) if groupnorm else nn.BatchNorm2d(96),
            activation(self.inplace),
            nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.GroupNorm(1, 96) if groupnorm else nn.BatchNorm2d(96),
            activation(self.inplace),
            nn.MaxPool2d(2),

            nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.GroupNorm(1, 96) if groupnorm else nn.BatchNorm2d(96),
            activation(self.inplace),
            nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.GroupNorm(1, 96) if groupnorm else nn.BatchNorm2d(96),
            activation(self.inplace),
            nn.MaxPool2d(2),

            nn.Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.GroupNorm(1, 96) if groupnorm else nn.BatchNorm2d(96),
            activation(self.inplace),
            Flatten(),
            nn.Linear(96 * 4 * 2, spec_out),
            nn.LayerNorm(spec_out) if groupnorm else nn.BatchNorm1d(spec_out),
            activation(self.inplace)])

        self.enc = nn.Sequential(*modules)

        self.kw, self.kh = 40, 78
        self.dw, self.dh = 1, 1

        self.pose = PositionalEncoding(d_model=32, max_len=60)
        self.seq_sa = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=32, nhead=8, dim_feedforward=256, dropout=0.1), num_layers=2)
        
        self.seq_model = nn.LSTM(spec_out, hidden_size=hidden_size, num_layers=n_lstm_layers, batch_first=True)
        
        self.z_enc = nn.Sequential(
            nn.Linear(hidden_size//2 + spec_out, zdim),
            nn.LayerNorm(zdim) if groupnorm else nn.BatchNorm1d(zdim),
            activation(self.inplace)
        )

        self.inference_x = deque(maxlen=self.kw)
        
        self.inference_seq = deque(maxlen=60)
        self.step_cnt = 0

    def forward(self, x, context):

        raise NotImplementedError

    def encode_sequence(self, x, hidden=None):

        x, last_steps, lengths = self.encode_samples(x)

        #  self-attention
        x_tmp = []
        for id, i in enumerate(x):
            # print(i.shape)
            padd = F.pad(i, (0, 0, 60-i.shape[0], 0))
            x_tmp.append(padd)
            # print(f"{id}/{len(x)}", i.shape)
        x = torch.stack(x_tmp).permute(1, 0, 2)

        x = self.pose(x)
        x = self.seq_sa(x)
        # use hidden state of last layer as conditioning information z
        z = self.z_enc(torch.cat((x[0, :, :], last_steps), -1))

        return z, hidden

    def encode_samples(self, x):

        last_steps = []

        zero_lengths = []
        for i in range(len(x)):

            if x[i].shape[0] < self.kw:
                padding = self.kw - x[i].shape[0]
                x[i] = F.pad(x[i], (0, 0, padding, 0), mode='constant')

            last_steps.append(x[i][-40:].unsqueeze(0))
            
            stacked = torch.stack(x[i][:self.kw * (x[i].shape[0]//self.kw)].split(self.kw)).unsqueeze(1)
            
            if stacked.shape[0] == 1:
                zero_lengths.append(i)
                x[i] = torch.zeros_like(stacked)
            else:
                x[i] = stacked

        lengths = [spec.shape[0] for spec in x]
        last_steps = self.enc(torch.stack(last_steps))
        x = self.enc(torch.cat(x))
        x = list(torch.split(x, lengths))

        for idx in zero_lengths:
            x[idx] = torch.zeros(1, x[idx].shape[-1], device=x[idx].device)
            lengths[idx] = 1

        return x, last_steps, lengths

    def get_conditioning(self, x, hidden=None):
        if hidden is None:
            self.inference_x = deque(maxlen=self.kw)
            self.inference_seq = deque(maxlen=60)
            
        self.inference_x.append(x)

        x = torch.cat(list(self.inference_x))
        
        if x.shape[0] < self.kw:
            padding = self.kw - x.shape[0]
            x = F.pad(x, (0, 0, padding, 0), mode='constant')
            
        last_steps = self.enc(x.unsqueeze(0).unsqueeze(0))
        
        if hidden is None:
            
            x = torch.zeros_like(last_steps)
            x = F.pad(x, (0, 0, 60-x.shape[0], 0))
               
                # print(f"{id}/{len(x)}", i.shape)
            x = x.unsqueeze(1)

            x = self.pose(x)
            hidden = self.seq_sa(x)
        # use hidden state of last layer as conditioning information z
        # print(x.shape, last_steps.shape)
        z = self.z_enc(torch.cat((hidden[0, :, :], last_steps), -1))

        self.step_cnt += 1
        if self.step_cnt == self.kw:

            # _, hidden = self.seq_model(last_steps.unsqueeze(0), hidden)
            self.inference_seq.append(last_steps)
            x = torch.cat(list(self.inference_seq))
            # print("x", x.shape)
            x = F.pad(x, (0, 0, 60-x.shape[0], 0))
               
            x = x.unsqueeze(1)

            x = self.pose(x)
            hidden = self.seq_sa(x)
            
            self.step_cnt = 0

        return z, hidden
  
import math
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0).transpose(0, 1)
        # print(pe.shape)

    def forward(self, x):
        # print(type(x), type(self.pe), x.shape, self.pe.shape)
        x = x + self.pe[:x.size(0), :].to(x.device)
        return self.dropout(x)