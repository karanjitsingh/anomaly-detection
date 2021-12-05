import copy
from torch.optim.adam import Adam
from preprocess_data import *
from utils import *
from config import *

import torch
from torch import nn

from datetime import datetime
# Third Party
import torch
import torch.nn as nn
from torchinfo import summary



class MLP_AE(nn.Module):


    def getEncoder(self, input_dim, out_dim, h_dims, h_activ, out_activ):
        layer_dims = [input_dim] + h_dims + [out_dim]

        layers = []
        num_layers = len(layer_dims) - 1
        for index in range(num_layers):
            layer = nn.Linear(layer_dims[index], layer_dims[index+1])
            layers.append(layer)
            layers.append(h_activ)


        return nn.Sequential(*layers)
        

    def getDecoder(self, input_dim, out_dim, h_dims, h_activ):

        layer_dims = [input_dim] + h_dims + [h_dims[-1]]
        num_layers = len(layer_dims) - 1
        layers = []
        for index in range(num_layers):
            layer = nn.Linear(layer_dims[index],layer_dims[index + 1])
            layers.append(layer)
            layers.append(h_activ)

        output_layer = nn.Linear(layer_dims[-1], out_dim)
        layers.append(output_layer)
        # layers.append(h_activ)

        return nn.Sequential(*layers)


    def __init__(self, input_dim, encoding_dim, h_dims=[], h_activ=nn.Tanh(),
                 out_activ=nn.Tanh(), config=Config()):
        super(MLP_AE, self).__init__()

        self.config: Config = config

        self.encoder = self.getEncoder(input_dim * config.len_seq, encoding_dim, h_dims, h_activ,
                               out_activ)
        self.decoder = self.getDecoder(encoding_dim, input_dim * config.len_seq, h_dims[::-1],
                               h_activ)


    def forward(self, x):
        batch_len, seq_len, f_n = x.shape
        x = x.reshape((batch_len,seq_len * f_n))

        x = self.encoder(x)
        x = self.decoder(x).reshape((batch_len, seq_len, f_n))

        return x

class TrainerGeneric():


    def print_stats(self):
        train_stats = np.unique([a for y in self.y_train for a in y],return_counts=True)[1]
        val_stats = np.unique([a for y in self.y_val for a in y],return_counts=True)[1]

        print('Training set statistics:')
        print(len(train_stats),'classes with distribution',train_stats)
        print('Validation set statistics:')
        print(len(val_stats),'classes with distribution',val_stats)

    def print_config(self):
        config = self.config
        print("Configuration:")
        vars = list(filter(lambda x: x[0] != '_', dir(config)) )
        for var in vars:
            print(str(var) + ": ", getattr(config, var))

    def print_model(self):
        # summary(self.net, input_size = (1000,self.config.len_seq * self.config.n_channels))
        pass

    def __init__(self, net, config:Config=Config())->None:
        self.config = config
        self.net = net


        self.train_on_gpu = config.train_on_gpu = train_on_gpu = torch.cuda.is_available()
        print("Train on GPU? ", train_on_gpu)


        self.X_train, self.y_train = load_data('train',config.len_seq,config.stride)
        self.X_val, self.y_val = load_data('val',config.len_seq,config.stride)

        net.apply(init_weights)


        self.optimizer = Adam(net.parameters(), lr=config.lr)
        # self.criterion = nn.L1Loss(reduction='sum')
        self.criterion = nn.MSELoss()
        
        self.print_model()

    def train(self, denoise=False):
        # criterion = MSELoss(size_average=False)
        optimizer = self.optimizer
        criterion = self.criterion

        config = self.config
        net = self.net

        if(config.train_on_gpu):
            net = net.cuda()

        mean_losses = []
        for epoch in range(1, config.num_epochs + 1):
            net.train()


            # # Reduces learning rate every 50 epochs
            # if not epoch % 50:
            #     for param_group in optimizer.param_groups:
            #         param_group["lr"] = lr * (0.993 ** epoch)

            losses = []

            # for batch in iterate_minibatches_2D(self.X_train, self.y_train, config.batch_size, config.stride, shuffle=True, num_batches=1, batchlen=config.batchlen, drop_last=True):
            for batch in iterate_minibatches_2D(self.X_train, self.y_train, config.batch_size, config.stride, shuffle=True, num_batches=config.num_batches, batchlen=config.batchlen, drop_last=True):
                # print("lol")
                x,y ,pos = batch
                # for x in seq:
                x = torch.from_numpy(x)

                if(config.train_on_gpu):
                    x = x.cuda()
                
                optimizer.zero_grad()

                # Forward pass
                x_prime = net(x)

                loss = criterion(x_prime, x)

                # Backward pass
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

            mean_loss = np.mean(losses)
            mean_losses.append(mean_loss)

            print(f"Epoch: {epoch}, Loss: {mean_loss}")

        return mean_losses


    def get_encodings(self, model, train_set):
        self.net.eval()
        encodings = [self.net.encoder(x) for x in train_set]
        return encodings


# class TrainSequiturLSTMAE(TrainerGeneric):
    
#     def __init__(self,encoding_dim=32, h_dims=[128], h_activ=nn.Sigmoid(), out_activ=nn.Tanh(), config: Config = Config()) -> None:
#         self.config = config
#         self.net = net = LSTM_AE(encoding_dim=encoding_dim, input_dim= config.n_channels, h_dims=h_dims, h_activ=h_activ,out_activ=out_activ, config=config)

#         super(TrainSequiturLSTMAE, self).__init__(net, config)
        



config:Config = Config()
config.batch_size = 1000
config.num_epochs = 20
config.lr = 0.001

mlp = MLP_AE(config.n_channels,32,[128,64],nn.Tanh(),nn.Tanh(),config)


trainer = TrainerGeneric(mlp,config=config)
losses = trainer.train()

# %matplotlib inline
import matplotlib.pyplot as plt
plt.plot(list(range(len(losses))), losses)
plt.show()
plt.yscale('log')
plt.plot(list(range(len(losses))), losses)
plt.show()
