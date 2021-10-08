from numpy import true_divide
from utils import *

def remove_anomalies(X,y, anom=1):
    indices = np.where(y==anom)
    intervals = []
    left = None
    right = None

    for i in list(indices[0]):
        if left is None:
            left = i
            right = i
        elif right + 1 == i:
            right = i
        else:
            intervals.append((left, right))
            left = None
            right = None

    if left is not None and intervals[-1] != (left, right):
        intervals.append((left, right))

    return intervals

def load_data(name,len_seq,stride, removeAnom = False):
    Xs = []
    ys = []

    ## Use glob module and wildcard to build a list of files to load from data directory
    path = "data/{}_data_*".format(name)
    data = glob.glob(path)

    for file in data:
        X, y = load_dataset(file)
        if removeAnom:
            ranges = remove_anomalies(X,y)
            print(ranges)
            last = 0
            for interval in ranges:
                Xi, yi = slide(X[last:interval[0]-1], y[last:interval[0]-1], len_seq, stride, save=False)
                last = interval[1]+1
                Xs.append(Xi)
                ys.append(yi)

            Xi, yi = slide(X[last:], y[last:], len_seq, stride, save=False)
            Xs.append(Xi)
            ys.append(yi)
        else:
            X, y = slide(X, y, len_seq, stride, save=False)
            Xs.append(X)
            ys.append(y)
        

    return Xs, ys


X,y = load_data('train',24,1, removeAnom=True)
X = np.array(X)
y = np.array(y)
for i in range(len(X)):
    print(y[i].shape)

print(len(X), len(y))

