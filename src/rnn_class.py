import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.autograd import Variable
from tqdm import tqdm
tqdm.pandas(desc='Progress')
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, roc_curve
#from robust_check_BD import train_test_split_BD
import json
from sklearn.preprocessing import StandardScaler
import numpy as np

def scaling(train, test, columns):
    #train_scaled = np.copy(train)
    #test_scaled = np.copy(test)
    scaler = StandardScaler()
    train[columns] = scaler.fit_transform(train[columns])
    test[columns] = scaler.transform(test[columns])
    return train, test

def sort_batch(X, y, lengths):
    lengths, indx = lengths.sort(dim=0, descending=True)
    X = X[indx]
    y = y[indx]
    return X.transpose(0,1), y, lengths # transpose (batch x seq) to (seq x batch)

class VectorizeData(Dataset):
    def __init__(self, ts_data, ts_times, ts_targets):
        self.ts_data = ts_data
        self.ts_times = ts_times
        self.ts_targets = ts_targets
        self.ts_padded_data = pad_sequence(self.ts_data, padding_value=0)

    def __len__(self):
        return len(self.ts_data)
    
    def __getitem__(self, idx):
        X = self.ts_padded_data[:,idx,:]
        lens = self.ts_data[idx].shape[0]
        y = self.ts_targets[idx]
        return X, y, lens
    
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size):
        super(LSTM, self).__init__()
        
        # Define the LSTM layer (bidirectional=True makes it bidirectional)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True)
        
        # Define the fully connected layer
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Multiply by 2 because of the bidirection

    def forward(self, x, lengths):
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.lstm.num_layers * 2, x.size(1), self.lstm.hidden_size).to(x.device)  # *2 for bidirection
        c0 = torch.zeros(self.lstm.num_layers * 2, x.size(1), self.lstm.hidden_size).to(x.device)

        packed_input = pack_padded_sequence(x, lengths)

        # Forward propagate through LSTM
        out, _ = self.lstm(packed_input, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size * 2)
        out, _ =  pad_packed_sequence(out)
        # Decode the hidden state of the last time step
        out = self.fc(out[lengths-1, range(x.size(1))])  # Only last time step is used for classification/regression
        return out

class build_rnn():

    def __init__(self, input_size, n_out, params):
        self.input_size = input_size
        self.n_out = n_out
        self.epochs = params['epochs']
        self.n_hidden = params['hidden size']
        self.lr = params['lr']
        self.n_layers = params['n_layers']
        self.dropout = params['dropout']
        self.opt_name = params['optimizer']

    def get_model(self, network_type = 'bilstm'):
        if network_type == 'bilstm':
            model = LSTM(self.input_size, self.n_hidden, self.n_layers, self.dropout, self.n_out).cpu()
        return model

    def fit(self, model, train_dl, val_dl):
        target_specificity = 0.8
        for epoch in range(self.epochs):
            if val_dl:
                y_true_val = list()
                pred_prob_val = list()
            if self.opt_name == 'adam':
                opt = optim.Adam(model.parameters(), self.lr)
            for X, y, lengths in train_dl:
                X, y,lengths = sort_batch(X,y,lengths)
                X = Variable(X)
                y = Variable(y)
                opt.zero_grad()
                pred = model(X, lengths)
                loss = F.cross_entropy(pred, y)
                loss.backward()
                opt.step()
            if val_dl:
                for X, y, lengths in val_dl:
                    X, y,lengths = sort_batch(X,y,lengths)
                    X = Variable(X)
                    y = Variable(y)
                    pred = model(X, lengths)
                    y_true_val += list(y.cpu().data.numpy())
                    pred_prob_val += list(pred[:,1].cpu().data.numpy())
                valroc = roc_auc_score(y_true_val, pred_prob_val)
                #print(f'Val roc auc: {valroc}')
                fpr, tpr, thresholds = roc_curve(y_true_val, pred_prob_val)
                specificity = 1 - fpr
                # Find the closest specificity and the corresponding sensitivity
                idx = np.argmin(np.abs(specificity - target_specificity))
                target_sensitivity = tpr[idx]

        return valroc, target_sensitivity
