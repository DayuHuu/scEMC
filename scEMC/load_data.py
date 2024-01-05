import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import h5py
import warnings
warnings.filterwarnings("ignore")



ALL_data = dict(
    #
    BMNC = {1: 'BMNC', 2: 'd1', 'N': 30672, 'K': 27, 'V': 2, 'n_input': [1000,25], 'n_hid': [10,256], 'n_output': 64},
    )


path = './datasets/'


def load_data(dataset):
    data = h5py.File(path + dataset[1] + ".mat")
    X = []
    Y = []
    Label = np.array(data['Y']).T
    Label = Label.reshape(Label.shape[0])
    mm = MinMaxScaler()
    for i in range(data['X'].shape[1]):
        diff_view = data[data['X'][0, i]]
        diff_view = np.array(diff_view, dtype=np.float32).T
        std_view = mm.fit_transform(diff_view)
        X.append(std_view)
        Y.append(Label)

    size = len(Y[0])
    view_num = len(X)

    index = [i for i in range(size)]
    np.random.shuffle(index)
    for v in range(view_num):
        X[v] = X[v][index]
        Y[v] = Y[v][index]

    for v in range(view_num):
        X[v] = torch.from_numpy(X[v])

    return X, Y





