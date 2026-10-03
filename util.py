import csv
import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
from torch.autograd import Variable


def mape_loss(target, input, mask=None):
    if mask is None:
        loss = torch.abs(input - target) / (torch.abs(target) + 1e-2)
        return loss.mean() * 100
    valid = mask.to(dtype=torch.bool) & torch.isfinite(target) & torch.isfinite(input)
    if not torch.any(valid):
        return input.sum() * 0.0
    # Sanitize masked entries before arithmetic so NaNs cannot poison gradients.
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    safe_input = torch.where(valid, input, torch.zeros_like(input))
    loss = torch.abs(safe_input - safe_target) / (torch.abs(safe_target) + 1e-2)
    return loss.masked_select(valid).mean() * 100


def MAPE(y_true, y_pre):
    y_true = (y_true).reshape((-1, 1))
    y_pre = (y_pre).reshape((-1, 1))

    # e = (y_true + y_pre) / 2 + 1e-2
    # re = (np.abs(y_true - y_pre) / (np.abs(y_true) + e)).mean()
    re = np.mean(np.abs((y_true - y_pre) / y_true)) * 100

    return re


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))


class DataLoaderS(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(
        self,
        file_name,
        train,
        valid,
        device,
        horizon,
        window,
        normalize=2,
        normalized_output_path=None,
        split_policy='legacy',
        materialize_test=True,
        label_path=None,
        label_mask_path=None,
    ):
        if split_policy not in {'legacy', 'embargo'}:
            raise ValueError(f"unknown split_policy {split_policy!r}")
        self.device = device
        self.P = window
        self.h = horizon
        self.normalized_output_path = normalized_output_path
        self.split_policy = split_policy
        self.materialize_test = bool(materialize_test)
        with open(file_name) as fin:
            self.rawdat = np.loadtxt(fin, delimiter=',', skiprows=1)
        if self.rawdat.ndim == 1:
            self.rawdat = self.rawdat.reshape(1, -1)
        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape
        base_dir = os.path.dirname(os.fspath(file_name))
        if label_path is None:
            candidate = os.path.join(base_dir, 'dataset_labels.csv')
            label_path = candidate if os.path.exists(candidate) else None
        if label_mask_path is None:
            candidate = os.path.join(base_dir, 'label_valid_mask.csv')
            label_mask_path = candidate if os.path.exists(candidate) else None
        train_end = int(train * self.n)
        valid_end = int((train + valid) * self.n)
        # Validation-only runs must not read future labels.  Keep a full-size
        # sidecar-shaped array so the existing indexing code remains stable;
        # rows beyond the validation boundary stay NaN/False and are never
        # materialized into a test tensor.
        label_limit = self.n if self.materialize_test else valid_end
        if label_path is not None:
            self.label_path = os.fspath(label_path)
            self.raw_labels = self._load_sidecar_prefix(label_path, self.n, 3, label_limit)
        else:
            self.label_path = None
            self.raw_labels = self.rawdat[:, :3].copy()
        if label_mask_path is not None:
            mask_values = self._load_sidecar_prefix(label_mask_path, self.n, 3, label_limit)
            self.label_valid_mask = np.isfinite(mask_values) & (mask_values != 0)
        else:
            self.label_valid_mask = np.isfinite(self.raw_labels)
        if label_path is not None and label_mask_path is not None:
            observed = np.arange(self.n) < label_limit
            if np.any(self.label_valid_mask[observed] & ~np.isfinite(self.raw_labels[observed])):
                raise ValueError('label_valid_mask.csv marks a non-finite loaded label as valid')
        elif np.any(self.label_valid_mask & ~np.isfinite(self.raw_labels)):
            raise ValueError('label_valid_mask.csv marks a non-finite dataset label as valid')
        self.label_dat = np.full((self.n, 3), np.nan, dtype=np.float64)
        self.scale_mean = np.ones(self.m)
        self.scale_std = np.ones(self.m)
        self.train_size = train
        # self.scale = np.ones(self.m)
        # self._absolute_distance_normalized(normalize)
        self._z_score_normalized(normalize)
        self.train_feas = self.dat[:int(train * self.n), :]
        self._split(int(train * self.n), int((train + valid) * self.n), self.n)

    @staticmethod
    def _load_sidecar_prefix(path, n_rows, n_cols, limit_rows):
        """Load only the prefix needed by a validation-only run.

        The returned array is always ``(n_rows, n_cols)``.  Unread future rows
        remain NaN, which prevents accidental test-label access while keeping
        the downstream batchification contract unchanged.
        """
        limit_rows = max(0, min(int(limit_rows), int(n_rows)))
        values = np.full((int(n_rows), int(n_cols)), np.nan, dtype=np.float64)
        if limit_rows == 0:
            return values
        with open(path, newline='') as fin:
            loaded = np.loadtxt(fin, delimiter=',', skiprows=1, max_rows=limit_rows)
        if loaded.ndim == 1:
            loaded = loaded.reshape(1, -1)
        if loaded.shape != (limit_rows, n_cols):
            raise ValueError(f'sidecar prefix must have shape {(limit_rows, n_cols)}, got {loaded.shape}')
        values[:limit_rows] = loaded
        return values


        # self.de_scale_std = torch.from_numpy(self.scale_std[:3]).float().to(self.device)
        # self.de_scale_mean = torch.from_numpy(self.scale_mean[:3]).float().to(self.device)

        # self.scale = torch.from_numpy(self.scale[:3]).float()
        # self.scale = self.scale.to(device)
        # self.scale = Variable(self.scale)

    def _z_score_normalized(self, normalize):
        for i in range(self.m):
            source = self.raw_labels[:, i] if i < 3 else self.rawdat[:, i]
            train_values = source[:int(self.train_size * self.n)]
            train_values = train_values[np.isfinite(train_values)]
            if len(train_values) == 0:
                raise ValueError(f'no finite training values available for column {i}')
            self.scale_mean[i] = np.mean(train_values)
            self.scale_std[i] = np.std(train_values)
            if self.scale_std[i] < 1e-8:
                self.scale_std[i] = 1.0
            self.dat[:, i] = (self.rawdat[:, i] - self.scale_mean[i]) / self.scale_std[i]
            if i < 3:
                self.label_dat[:, i] = (self.raw_labels[:, i] - self.scale_mean[i]) / self.scale_std[i]
        if self.normalized_output_path is not None:
            output_path = os.fspath(self.normalized_output_path)
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            header = ','.join(str(i) for i in range(self.m))
            np.savetxt(output_path, self.dat, delimiter=',', header=header, comments='')

    def _de_z_score_normalized(self, y, device_flag):
        return self._denorm_cols(y, [0, 1, 2], device_flag)

    def _denorm_cols(self, y, cols, device_flag):
        cols = np.asarray(cols, dtype=int)
        if device_flag == 'cpu':
            de_scale_std = self.scale_std[cols]
            de_scale_mean = self.scale_mean[cols]
            return y * de_scale_std + de_scale_mean
        else:
            de_scale_std = torch.from_numpy(self.scale_std[cols]).float().to(self.device)
            de_scale_mean = torch.from_numpy(self.scale_mean[cols]).float().to(self.device)
            return y * de_scale_std + de_scale_mean

    def _absolute_distance_normalized(self, normalize):
        for i in range(self.m):
            self.scale[i] = np.max(np.abs(self.rawdat[:, i]))
            self.dat[:, i] = self.rawdat[:, i] / np.max(np.abs(self.rawdat[:, i]))


    def _split(self, train, valid, test):
        train_set = range(self.P + self.h - 1, train)
        if self.split_policy == 'embargo':
            valid_start = train + self.P + self.h
            test_start = valid + self.P + self.h
        else:
            valid_start = train
            test_start = valid
        valid_set = range(valid_start, valid)
        test_set = range(test_start, test)
        if len(train_set) == 0 or len(valid_set) == 0:
            raise ValueError(
                f"split_policy={self.split_policy!r} leaves an empty train or validation split"
            )
        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.test = self._batchify(test_set, self.h) if self.materialize_test else None
        self.train_label_mask = self._batchify_mask(train_set, self.h)
        self.valid_label_mask = self._batchify_mask(valid_set, self.h)
        self.test_label_mask = self._batchify_mask(test_set, self.h) if self.materialize_test else None
        self.split_metadata = {
            'split_policy': self.split_policy,
            'window_h': int(self.P),
            'horizon_h': int(self.h),
            'embargo_h': int(self.P + self.h) if self.split_policy == 'embargo' else 0,
            'boundaries': {
                'train_end_exclusive': int(train),
                'validation_end_exclusive': int(valid),
                'data_end_exclusive': int(test),
            },
            'sample_counts': {
                'train': len(train_set),
                'validation': len(valid_set),
                'test_available': len(test_set),
            },
            'target_end_index_ranges': {
                'train': [train_set.start, train_set.stop - 1],
                'validation': [valid_set.start, valid_set.stop - 1],
                'test': [test_set.start, test_set.stop - 1],
            },
            'test_materialized': self.test is not None,
            'label_sidecars': {
                'path': getattr(self, 'label_path', None),
                'loaded_through_row_exclusive': int(test if self.materialize_test else valid),
                'valid_counts': {
                    'train': int(self.train_label_mask.sum()),
                    'validation': int(self.valid_label_mask.sum()),
                    'test_available': None if self.test_label_mask is None else int(self.test_label_mask.sum()),
                },
                'invalid_counts': {
                    'train': int((~self.train_label_mask).sum()),
                    'validation': int((~self.valid_label_mask).sum()),
                    'test_available': None if self.test_label_mask is None else int((~self.test_label_mask).sum()),
                },
            },
        }

    def _batchify(self, idx_set, horizon):
        # print("datshape", self.dat.shape)
        n = len(idx_set)
        X = torch.zeros((n, self.P, self.m))
        Y = torch.zeros((n, self.h, self.m))
        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P

            X[i, :, :] = torch.from_numpy(self.dat[start:end, :])
            Y[i, :, :] = torch.from_numpy(self.dat[idx_set[i] + 1 - horizon:idx_set[i] + 1, :])
            Y[i, :, :3] = torch.from_numpy(self.label_dat[idx_set[i] + 1 - horizon:idx_set[i] + 1, :3])

        return [X, Y]

    def _batchify_mask(self, idx_set, horizon):
        n = len(idx_set)
        mask = np.zeros((n, self.h, 3), dtype=bool)
        for i in range(n):
            mask[i, :, :] = self.label_valid_mask[idx_set[i] + 1 - horizon:idx_set[i] + 1, :3]
        return torch.from_numpy(mask)

    def get_batches(self, inputs, targets, batch_size, shuffle=True, masks=None):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            X = X.to(self.device)
            Y = Y.to(self.device)
            if masks is None:
                yield Variable(X), Variable(Y)
            else:
                batch_mask = masks[excerpt].to(self.device)
                yield Variable(X), Variable(Y), Variable(batch_mask)
            start_idx += batch_size


class DataLoaderM(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()


class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


def asym_adj(adj):
    """Asymmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_adj(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj = load_pickle(pkl_filename)
    return adj


def load_dataset(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])

    data['train_loader'] = DataLoaderM(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoaderM(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoaderM(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler
    return data


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse


def load_node_feature(path):
    fi = open(path)
    x = []
    for li in fi:
        li = li.strip()
        li = li.split(",")
        e = [float(t) for t in li[1:]]
        x.append(e)
    x = np.array(x)
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    z = torch.tensor((x - mean) / std, dtype=torch.float)
    return z


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))
