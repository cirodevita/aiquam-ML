import sys
import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import squareform
from scipy.stats import mode
#from clang.cindex import xrange
from numpy import shape
from sklearn.neighbors import KNeighborsClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.knn = KNeighborsClassifier(n_neighbors=configs.n_neighbors)
        self.x = None
        self.y = None

    def fit(self, x, y):
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)
        self.knn.fit(self.x, self.y)

    def evaluate(self, x, y):
        y_pred = self.knn.predict(x)
        accuracy = np.sum(y_pred == y) / len(y)
        return accuracy
    
    def save_model(self, path):
        initial_type = [('input', FloatTensorType([None, self.x.shape[1]]))]
        onx = convert_sklearn(self.knn, initial_types=initial_type, target_opset=18)

        # onx = to_onnx(self.knn, self.x[:1])
        with open(f"{path}/model.onnx", "wb") as f:
            f.write(onx.SerializeToString())

"""
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.knn_dtw = KnnDtw(configs)

    def fit(self, x, y):
        self.knn_dtw.fit(x, y)

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # Convert input tensor to numpy array
        x_np = x.cpu().numpy()
        # Predict using the KNN model
        y_pred, _ = self.knn_dtw.predict(x_np)
        # Convert numpy array back to tensor
        y_pred_tensor = torch.from_numpy(y_pred).to(x.device)
        return y_pred_tensor

    def evaluate(self, x, y):
        return self.knn_dtw.evaluate(x, y)
    
    def save_model(self, path):
        torch.save(self.state_dict(), path + '/checkpoint.pth')
        self.knn_dtw.save_state(path + '/knn_dtw_state.npz')

    def load_model(self, path):
        self.load_state_dict(torch.load(path + '/checkpoint.pth'))
        self.knn_dtw.load_state(path + '/knn_dtw_state.npz')


class KnnDtw(object):
    def __init__(self, configs):
        self.n_neighbors = configs.n_neighbors
        self.max_warping_window = configs.max_warping_window
        self.subsample_step = configs.subsample_step
        self.x = None
        self.l = None

    def fit(self, x, l):
        self.x = x
        self.l = l

    def _dtw_distance(self, ts_a, ts_b, d=lambda x, y: abs(x - y)):
        # Create cost matrix via broadcasting with large int
        ts_a, ts_b = np.array(ts_a), np.array(ts_b)
        M, N = len(ts_a), len(ts_b)
        cost = sys.maxsize * np.ones((M, N))

        # Initialize the first row and column
        cost[0, 0] = d(ts_a[0], ts_b[0])
        for i in xrange(1, M):
            cost[i, 0] = cost[i - 1, 0] + d(ts_a[i], ts_b[0])

        for j in xrange(1, N):
            cost[0, j] = cost[0, j - 1] + d(ts_a[0], ts_b[j])

        # Populate rest of cost matrix within window
        for i in xrange(1, M):
            for j in xrange(max(1, i - self.max_warping_window),
                            min(N, i + self.max_warping_window)):
                choices = cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]
                cost[i, j] = min(choices) + d(ts_a[i], ts_b[j])

        # Return DTW distance given window
        return cost[-1, -1]

    def _dist_matrix(self, x, y):
        # Compute the distance matrix
        dm_count = 0

        # Compute condensed distance matrix (upper triangle) of pairwise dtw distances
        # when x and y are the same array
        if np.array_equal(x, y):
            x_s = np.shape(x)
            dm = np.zeros((x_s[0] * (x_s[0] - 1)) // 2, dtype=np.double)

            p = ProgressBar(shape(dm)[0])

            for i in xrange(0, x_s[0] - 1):
                for j in xrange(i + 1, x_s[0]):
                    dm[dm_count] = self._dtw_distance(x[i, ::self.subsample_step],
                                                      y[j, ::self.subsample_step])

                    dm_count += 1
                    p.animate(dm_count)

            # Convert to squareform
            dm = squareform(dm)
            return dm

        # Compute full distance matrix of dtw distnces between x and y
        else:
            x_s = np.shape(x)
            y_s = np.shape(y)
            dm = np.zeros((x_s[0], y_s[0]))
            dm_size = x_s[0] * y_s[0]

            p = ProgressBar(dm_size)

            for i in xrange(0, x_s[0]):
                for j in xrange(0, y_s[0]):
                    dm[i, j] = self._dtw_distance(x[i, ::self.subsample_step],
                                                  y[j, ::self.subsample_step])
                    # Update progress bar
                    dm_count += 1
                    p.animate(dm_count)

            return dm

    def predict(self, x):
        dm = self._dist_matrix(x, self.x)

        # Identify the k nearest neighbors
        knn_idx = dm.argsort()[:, :self.n_neighbors]

        # Identify k nearest labels
        knn_labels = self.l[knn_idx]

        # Model Label
        mode_data = mode(knn_labels, axis=1)
        mode_label = mode_data[0]
        mode_proba = mode_data[1] / self.n_neighbors

        return mode_label.ravel(), mode_proba.ravel()
    
    def evaluate(self, x, y):
        y_pred, _ = self.predict(x)
        accuracy = np.sum(y_pred == y) / len(y)
        return accuracy
    
    def save_state(self, path):
        np.savez(path, x=self.x, l=self.l)

    def load_state(self, path):
        data = np.load(path)
        self.x = data['x']
        self.l = data['l']


class ProgressBar:
    def __init__(self, iterations):
        self.iterations = iterations
        self.prog_bar = '[]'
        self.fill_char = '*'
        self.width = 40
        self.__update_amount(0)
        self.animate = self.animate_ipython

    def animate_ipython(self, iter):
        print('\r', self, sys.stdout.flush())
        self.update_iteration(iter + 1)

    def update_iteration(self, elapsed_iter):
        self.__update_amount((elapsed_iter / float(self.iterations)) * 100.0)
        self.prog_bar += '  %d of %s complete' % (elapsed_iter, self.iterations)

    def __update_amount(self, new_amount):
        percent_done = int(round((new_amount / 100.0) * 100.0))
        all_full = self.width - 2
        num_hashes = int(round((percent_done / 100.0) * all_full))
        self.prog_bar = '[' + self.fill_char * num_hashes + ' ' * (all_full - num_hashes) + ']'
        pct_place = (len(self.prog_bar) // 2) - len(str(percent_done))
        pct_string = '%d%%' % percent_done
        self.prog_bar = self.prog_bar[0:pct_place] + \
                        (pct_string + self.prog_bar[pct_place + len(pct_string):])

    def __str__(self):
        return str(self.prog_bar)
"""