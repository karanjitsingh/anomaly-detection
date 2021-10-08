from preprocess_data import *
from utils import *
from config import *

import torch
from torch import nn

## Define our DeepConvLSTM class, subclassing nn.Module.
class DeepConvLSTM(nn.Module):

    def __init__(self, config, n_hidden = 128, n_layers = 2, n_filters = 64,
                n_classes = 18, filter_size = 5,pool_filter_size=3, drop_prob = 0.5, anom_threshold=0.5):

        super(DeepConvLSTM, self).__init__() # Call init function for nn.Module whenever this function is called

        self.drop_prob = drop_prob # Dropout probability
        self.n_layers = n_layers # Number of layers in the lstm network
        self.n_hidden = n_hidden # number of hidden units per layer in the lstm
        self.n_filters = n_filters # number of convolutional filters per layer
        self.n_classes = n_classes # number of target classes
        self.filter_size = filter_size # convolutional filter size
        self.pool_filter_size = pool_filter_size # max pool filter size if using
        self.config = config

        # Convolutional net
        self.convlayer = nn.Sequential(
            nn.Conv1d(config.n_channels, n_filters, (filter_size)),
            # nn.MaxPool2d((pool_filter_size,1)), # Max pool layers, optional. 
            nn.Conv1d(n_filters, n_filters, (filter_size)),
            # nn.MaxPool2d((pool_filter_size,1)),
            nn.Conv1d(n_filters, n_filters, (filter_size)),
            nn.Conv1d(n_filters, n_filters, (filter_size))
            )

        # LSTM layers
        self.lstm = nn.LSTM(n_filters, n_hidden, n_layers, batch_first=True)

        # Dropout layer
        self.dropout = nn.Dropout(p=drop_prob)

        # Output layer

        # Let's change this to 1 class with threshold 0.3
        # self.predictor = nn.Linear(n_hidden,n_classes)


        self.linear = nn.Linear(n_hidden,1)
        self.predictor = nn.Sigmoid()


    def forward(self, x, hidden, batch_size):

        #Reshape x if necessary to add the 2nd dimension
        x = x.view(-1, self.config.n_channels, self.config.len_seq)
        x = self.convlayer(x)
        x = x.view(self.config.batch_size, -1, self.n_filters)

        x,hidden = self.lstm(x, hidden)

        x = self.dropout(x)

        x = x.view(batch_size, -1, self.n_hidden)[:,-1,:]
        linout = self.linear(x)

        out = self.predictor(linout)

        return out, hidden

    def init_hidden(self, batch_size):
        config = self.config
        weight = next(self.parameters()).data # return a Tensor from self.parameters to use as a base for the initial hidden state.
        
        ## Generate new tensors of zeros with similar type to weight, but different size.
        if (config.train_on_gpu):
            hidden = (weight.new_zeros(self.n_layers, batch_size, self.n_hidden).cuda(), # Hidden state
                  weight.new_zeros(self.n_layers, batch_size, self.n_hidden).cuda()) # Cell state
        else:
            hidden = (weight.new_zeros(self.n_layers, batch_size, self.n_hidden),
                      weight.new_zeros(self.n_layers, batch_size, self.n_hidden))

        return hidden

from datetime import datetime
import sklearn.metrics as metrics
import csv

class TrainDeepConvLSTM():
    net = None

    def __init__(self, config = Config()):

        self.config = config
        self.train_on_gpu = config.train_on_gpu = train_on_gpu = torch.cuda.is_available()
        print("Train on GPU? ", train_on_gpu)

        self.net = net = DeepConvLSTM(n_classes = len(class_names), config = config)

        self.X_train, self.y_train = load_data('train',config.len_seq,config.stride)
        self.X_val, self.y_val = load_data('val',config.len_seq,config.stride)

        net.apply(init_weights)

        self.weight_decay = weight_decay = 1e-5*config.lr*config.batch_size*(50/config.batchlen)
        self.opt = opt = torch.optim.Adam(net.parameters(),lr=config.lr,weight_decay=weight_decay,amsgrad=True)
        self.scheduler = scheduler = torch.optim.lr_scheduler.StepLR(opt,100) # Learning rate scheduler to reduce LR every 100 epochs

        if(train_on_gpu):
            net.cuda()

        train_stats = np.unique([a for y in self.y_train for a in y],return_counts=True)[1]
        val_stats = np.unique([a for y in self.y_val for a in y],return_counts=True)[1]

        print('Training set statistics:')
        print(len(train_stats),'classes with distribution',train_stats)
        print('Validation set statistics:')
        print(len(val_stats),'classes with distribution',val_stats)

        self.weights = weights = torch.tensor([max(train_stats)/i for i in train_stats])

        if train_on_gpu:
            weights = weights.cuda()
            

        self.criterion = nn.CrossEntropyLoss(weight=weights)
        self.val_criterion = nn.CrossEntropyLoss()

        # Since we're gonna use single neuron, we will use a scalar loss
        self.criterion = nn.HuberLoss()
        self.val_criterion = nn.HuberLoss()

        self.early_stopping = EarlyStopping(patience=config.patience, verbose=False)

    def train(self):
        config = self.config
        net = self.net
        print('Starting training at',datetime.now())
        self.start_time=datetime.now()


        with open('log.csv', 'w', newline='') as csvfile: # We will save some training statistics to plot a loss curve later.

            for e in range(config.num_epochs):

                train_losses = []
                net.train() # Setup network for training

                for batch in iterate_minibatches_2D(self.X_train, self.y_train, config.batch_size, config.stride, shuffle=True, num_batches=config.num_batches, batchlen=config.batchlen, drop_last=True):

                    x,y,pos= batch

                    inputs, targets = torch.from_numpy(x), torch.from_numpy(y) # Get torch tensors.


                    self.opt.zero_grad() # Clear gradients in optimizer

                    if pos==0:
                        h = net.init_hidden(inputs.size()[0]) # If we are at the beginning of a metabatch, init lstm hidden states.

                        
                    h = tuple([each.data for each in h])  # Get rid of gradients attached to hidden and cell states of the LSTM
                    
                    if self.train_on_gpu:
                        inputs,targets = inputs.cuda(),targets.cuda()
                        


                    output, h = net(inputs,h,inputs.size()[0]) # Run inputs through network
                    # print(targets.unsqueeze(1).long().shape, output.double().shape)

                    loss = self.criterion(output.double(), targets.unsqueeze(1).double()) 
                    loss.backward()
                    self.opt.step()

                    train_losses.append(loss.item())


                val_losses = []
                net.eval() # Setup network for evaluation

                top_classes = []
                targets_cumulative = []

                
                with torch.no_grad():
                    for batch in iterate_minibatches_2D(self.X_val, self.y_val, config.val_batch_size, config.stride, shuffle=True, num_batches=config.num_batches_val, batchlen=config.batchlen, drop_last=False):

                        x,y,pos=batch


                        inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

                        targets_cumulative.extend([y for y in y])

                        if pos == 0:
                            val_h = net.init_hidden(inputs.size()[0]) # Init lstm at start of each metabatch

                        if self.train_on_gpu:
                            inputs,targets = inputs.cuda(),targets.cuda()

                        output, val_h = net(inputs,val_h,inputs.size()[0])

                        val_loss = self.val_criterion(output, targets.unsqueeze(1).long())
                        val_losses.append(val_loss.item())

                        top_p, top_class = output.topk(1,dim=1)
                        top_classes.extend([top_class.item() for top_class in top_class.cpu()])

                equals = [top_classes[i] == target for i,target in enumerate(targets_cumulative)]
                accuracy = np.mean(equals)

                f1score = metrics.f1_score(targets_cumulative, top_classes, average='weighted')
                f1macro = metrics.f1_score(targets_cumulative, top_classes, average='macro')

                self.scheduler.step()

                print('Epoch {}/{}, Train loss: {:.4f}, Val loss: {:.4f}, Acc: {:.2f}, f1: {:.2f}, Macro f1: {:.2f}'.format(e+1,config.num_epochs,np.mean(train_losses),np.mean(val_losses),accuracy,f1score,f1macro))


                writer = csv.writer(csvfile, delimiter=' ',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([np.mean(train_losses),np.mean(val_losses),accuracy,f1score,f1macro])
                
                self.early_stopping(np.mean(val_losses), net)
                if self.early_stopping.early_stop:
                    print("Stopping training, validation loss has not decreased in {} epochs.".format(config.patience))
                    break

                print('Training finished at ',datetime.now())
                print('Total time elapsed during training:',(datetime.now()-self.start_time).total_seconds(),'seconds')


    def test(self):


        # Commented out IPython magic to ensure Python compatibility.
        ### Test the model
        # %matplotlib inline
        config = self.config
        net = self.net
        from sklearn.metrics import classification_report,confusion_matrix
        import pandas as pd
        import seaborn as sn
        import matplotlib.pyplot as plt

        X_test, y_test = load_data('test',config.len_seq,config.stride)

        print('Starting testing at', datetime.now())
        self.start_time=datetime.now()
        # criterion = nn.CrossEntropyLoss()
        criterion = nn.HuberLoss()


        if(self.train_on_gpu):
            net.cuda()

        net.eval()

        val_losses = []
        accuracy=0
        f1score=0
        f1macro=0
        targets_cumulative = []
        top_classes = []

        i=0

        self.targets_all = targets_all = np.array([])
        self.outputs_all = outputs_all = np.array([])
        with torch.no_grad():

            for batch in iterate_minibatches_test(X_test, y_test, config.len_seq, config.stride):
                i+=1


                x,y,pos=batch



                inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

                targets_cumulative.extend([y for y in y])




                if(self.train_on_gpu):
                    targets,inputs = targets.cuda(),inputs.cuda()

                if pos == 0:
                    test_h = net.init_hidden(inputs.size()[0])


                output, test_h = net(inputs,test_h,inputs.size()[0])
                # print(inputs[0, :, 2])
                # print(inputs[1, :, 2])

                # print(targets)
                # print(inputs.shape, output)

                val_loss = criterion(output, targets.long())
                val_losses.append(val_loss.item())

                # top_p, top_class = output.topk(1,dim=1)
                # top_classes.extend([p.item() for p in top_class])

                targets_all = np.hstack([targets_all, targets.cpu().numpy()])
                # print(output.cpu().numpy()[:,0])
                outputs_all = np.hstack([outputs_all, output.cpu().numpy()[:,0]])

                # break


    def plot(self):
        config = self.config
        # Commented out IPython magic to ensure Python compatibility.
        print(self.targets_all.shape, self.outputs_all.shape)

        import matplotlib.pyplot as plt
        # %matplotlib inline
        # %matplotlib widget

        tx = range(len(self.targets_all))

        plt.figure(figsize=(15,3))
        plt.plot(tx, self.targets_all)
        plt.plot(tx, self.outputs_all)
        plt.show()
        # plot_data()

        i = 0
        print(config.num_batches)
        for batch in iterate_minibatches_2D(self.X_train, self.y_train, config.batch_size, config.stride, shuffle=False, num_batches=num_batches, batchlen=batchlen, drop_last=True):
            i +=1
            print(i, batch[0].shape)
            print(i, batch[1].shape)
            print(batch[0])
            if(i==1):
                break;

        print(self.X_test[0].shape, self.y_test[0].shape)


config = Config()
config.num_epochs = 1
dcl = TrainDeepConvLSTM(config = config)
dcl.train()
dcl.test()
dcl.plot()