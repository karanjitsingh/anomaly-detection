class_names = ['No Freeze','Freeze']

class Config():
    n_channels = 9 # number of sensor channels
    len_seq = 24 # Sliding window length
    stride = 1 # Sliding window step
    num_epochs = 20 # Max no. of epochs to train for
    num_batches= 20 # No. of training batches per epoch. -1 means all windows will be presented at least once, up to batchlen times per epoch (unless undersampled)
    batch_size = 1000 # Batch size / width - this many windows of data will be processed at once
    patience= 20 # Patience of early stopping routine. If criteria does not decrease in this many epochs, training is stopped.
    batchlen = 50 # No. of consecutive windows in a batch. If false, the largest number of windows possible is used.
    val_batch_size = 1000 # Batch size for validation/testing. 
    test_batch_size = 10000 # Useful to make this as large as possible given GPU memory, to speed up testing.
    lr = 0.0001 # Initial (max) learning rate
    num_batches_val = 1 # How many batches should we validate on each epoch
    lr_step = 100