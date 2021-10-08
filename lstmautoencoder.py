import copy
from torch.optim.adam import Adam
from preprocess_data import *
from utils import *
from config import *

import torch
from torch import nn

from datetime import datetime

class LSTMAutoEncoder(nn.Module):

    def __init__(self, config: Config, n_hidden=128, n_layers=1, n_code=64, n_code_layers=1):
        super(LSTMAutoEncoder, self).__init__()
        self.config = config
        self.n_hidden = n_hidden
        self.n_code = n_code
        self.n_layers = n_layers
        self.n_code_layers = n_code_layers


        # Encoder
        self.lstm1 = nn.LSTM(config.n_channels, self.n_hidden, self.n_layers, True)
        self.lstm2 = nn.LSTM(self.n_hidden, self.n_code, self.n_code_layers, True)

        # Decoder
        self.lstm3 = nn.LSTM(self.n_code, self.n_code, self.n_code_layers, True)
        self.lstm4 = nn.LSTM(self.n_code, self.n_hidden, self.n_layers, True)


        self.output_layer = nn.Linear(self.n_hidden, config.n_channels)

    def forward(self, x, hidden, batch_size):
        config = self.config

        x = x.reshape((-1, config.len_seq, config.n_channels))
        x, hidden[0] = self.lstm1(x, hidden[0])
        x, hidden[1] = self.lstm2(x, hidden[1])
        
        # x = hidden[1][0].reshape((config.n_channels, self.n_code))

        x = x.repeat(1,config.len_seq, config.n_channels)

        x, hidden[2] = self.lstm3(x, hidden[2])
        x, hidden[3] = self.lstm4(x, hidden[3])
        # x = x.reshape((config.len_seq, self.n_hidden))

        return self.output_layer(x), hidden

    
    def init_hidden(self, batch_size):
        config = self.config
        weight = next(self.parameters()).data # return a Tensor from self.parameters to use as a base for the initial hidden state.
        
        hidden = [None, None, None, None]

        ## Generate new tensors of zeros with similar type to weight, but different size.
        if (config.train_on_gpu):
            hidden[0] = (weight.new_zeros(self.n_layers, config.len_seq, self.n_hidden).cuda(), # Hidden state
                         weight.new_zeros(self.n_layers, config.len_seq, self.n_hidden).cuda()) # Cell state
            hidden[1] = (weight.new_zeros(self.n_code_layers, config.len_seq, self.n_code).cuda(), # Hidden state
                         weight.new_zeros(self.n_code_layers, config.len_seq, self.n_code).cuda()) # Cell state
            hidden[2] = (weight.new_zeros(self.n_code_layers, config.len_seq, self.n_code).cuda(), # Hidden state
                         weight.new_zeros(self.n_code_layers, config.len_seq, self.n_code).cuda()) # Cell state
            hidden[3] = (weight.new_zeros(self.n_layers, config.len_seq, self.n_hidden).cuda(), # Hidden state
                         weight.new_zeros(self.n_layers, config.len_seq, self.n_hidden).cuda()) # Cell state
                  
        else:
            hidden[0] = (weight.new_zeros(self.n_layers, config.len_seq, self.n_hidden), # Hidden state
                         weight.new_zeros(self.n_layers, config.len_seq, self.n_hidden)) # Cell state
            hidden[1] = (weight.new_zeros(self.n_code_layers, config.len_seq, self.n_code), # Hidden state
                         weight.new_zeros(self.n_code_layers, config.len_seq, self.n_code)) # Cell state
            hidden[2] = (weight.new_zeros(self.n_code_layers, config.len_seq, self.n_code), # Hidden state
                         weight.new_zeros(self.n_code_layers, config.len_seq, self.n_code)) # Cell state
            hidden[3] = (weight.new_zeros(self.n_layers, config.len_seq, self.n_hidden), # Hidden state
                         weight.new_zeros(self.n_layers, config.len_seq, self.n_hidden)) # Cell state
                  
        return hidden


class TrainLSTMAutoEncoder():
    net = None
    config: Config = None
    
    def print_stats(self):
        train_stats = np.unique([a for y in self.y_train for a in y],return_counts=True)[1]
        val_stats = np.unique([a for y in self.y_val for a in y],return_counts=True)[1]

        print('Training set statistics:')
        print(len(train_stats),'classes with distribution',train_stats)
        print('Validation set statistics:')
        print(len(val_stats),'classes with distribution',val_stats)


    def __init__(self, config = Config()) -> None:
        self.config = config
        self.train_on_gpu = config.train_on_gpu = train_on_gpu = torch.cuda.is_available()
        print("Train on GPU? ", train_on_gpu)

        self.net = net = LSTMAutoEncoder(config = config)


        self.X_train, self.y_train = load_data('train',config.len_seq,config.stride)
        self.X_val, self.y_val = load_data('val',config.len_seq,config.stride)

        net.apply(init_weights)


        self.optimizer = Adam(net.parameters(), lr=config.lr)
        self.criterion = nn.L1Loss(reduction='sum')
        self.history = dict(train=[], val=[])
        self.best_model_wts = copy.deepcopy(net.state_dict)
        self.best_loss = 10000.0
        # self.print_stats()

    def train(self):
        config = self.config
        optimizer = self.optimizer
        criterion = self.criterion
        history = self.history
        best_model_wts = self.best_model_wts
        best_loss = self.best_loss
        net = self.net

        print('Starting training at',datetime.now())
        self.start_time=datetime.now()

        for e in range(config.num_epochs):
            train_losses = []
            net.train()


            # batches = iterate_minibatches_2D(
            #     self.X_train,
            #     self.y_train,
            #     config.batch_size,
            #     config.stride,
            #     shuffle=True,
            #     num_batches=config.num_batches,
            #     batchlen=config.batchlen,
            #     drop_last=True)

            batches = iterate_minibatches_2D(
                self.X_train,
                self.y_train,
                config.batch_size,
                config.len_seq,
                shuffle=True,
                num_batches=config.num_batches,
                batchlen=1,
                drop_last=True)

            for batch in batches:
                optimizer.zero_grad() # Clear gradients in optimizer
                x,y,pos= batch

                inputs, targets = torch.from_numpy(x), torch.from_numpy(y) # Get torch tensors.

                if pos==0:
                    h = net.init_hidden(inputs.size()[0]) # If we are at the beginning of a metabatch, init lstm hidden states.
                    
                h = [(tuple([each.data for each in hx])) for hx in h]
                    
                if self.train_on_gpu:
                    inputs,targets = inputs.cuda(),targets.cuda()
                    
                # why
                # Get rid of gradients attached to hidden and cell states of the LSTM


                output, h = net(inputs, h, inputs.size()[0])
                
                loss = criterion(output,inputs)
                loss.backward()
                optimizer.step()




config = Config()
config.lr = 1e-3
config.num_epochs = 1
trainer = TrainLSTMAutoEncoder()
trainer.train()