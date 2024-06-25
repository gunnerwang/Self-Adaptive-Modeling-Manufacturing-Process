import random
from config import *
import numpy as np
from numpy import *
from collections import defaultdict
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, zscore
from tslearn.metrics import dtw, dtw_path
import torchdrift
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt


def visualize(y_pred: np.ndarray, y_true: np.ndarray, **kwargs):
    if not seq2seq_learning:
        y_pred_local = y_pred
        y_true_local = y_true
    else: 
        y_pred_local = y_pred[:, -1]
        y_true_local = y_true[:, -1]

    task_mask = [0] * len(y_true_local)
    task_idx, epoch_idx = None, None
    met_values = defaultdict(float)
    if 'task_mask' in kwargs.keys() and kwargs['task_mask'] is not None: task_mask = kwargs['task_mask']
    if 'task_idx' in kwargs.keys(): task_idx = kwargs['task_idx']
    if 'epoch_idx' in kwargs.keys(): epoch_idx = kwargs['epoch_idx']
    if 'metrics' in kwargs.keys() and kwargs['metrics'] is not None: met_values = kwargs['metrics']

    plt.figure()
    plt.subplot(211)
    plt.scatter(range(len(y_true_local)), y_true_local, c=task_mask, s=10)
    plt.plot(range(len(y_true_local)), y_true_local, label='true')
    plt.scatter(range(len(y_pred_local)), y_pred_local, s=10)
    plt.plot(range(len(y_pred_local)), y_pred_local, label='predicted')
    plt.legend()
    plt.title('Ground-truth vs. Prediction.')

    plt.subplot(212)
    plt.scatter(range(len(y_pred_local)), abs(y_pred_local - y_true_local))
    plt.plot(range(len(y_pred_local)), abs(y_pred_local - y_true_local))
    plt.title('L1 Prediction Error.')

    plt.grid()
    plt.savefig('./plots/{}-{}-{:.2f}-{:.2f}.png'.format(task_idx, epoch_idx, met_values['R2'], met_values['PCC']))

    if seq2seq_learning:
        fig, axs = plt.subplots(3,3, figsize=(15, 6), facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace = .5, wspace=.001)
        axs = axs.ravel()
        for i in range(9):
            cur_idx = random.randint(0, y_pred.shape[0]-1)
            axs[i].scatter(range(y_true.shape[1]), y_true[cur_idx, :], s=10)
            axs[i].plot(y_true[cur_idx, :], label='true')
            axs[i].scatter(range(y_pred.shape[1]), y_pred[cur_idx, :], s=10)
            axs[i].plot(y_pred[cur_idx, :], label='predicted')
            axs[i].grid()
            axs[i].legend()
            axs[i].set_title('Sequence {}'.format(cur_idx))
        plt.savefig('./seq_plots/{}-{}-{:.2f}-{:.2f}.png'.format(task_idx, epoch_idx, met_values['R2'], met_values['PCC']))

'''
perform the inter-sample evaluation
input: y_pred and y_true contain the samples
'''
def metrics_inter(y_pred: np.ndarray, y_true: np.ndarray):
    fitting_error = abs(y_pred - y_true)
    MAE = sum(fitting_error) / float(len(fitting_error))
    MSE = sum([item**2 for item in fitting_error]) / float(len(fitting_error))
    pearson_r, _ = pearsonr(y_pred, y_true)
    R2 = r2_score(y_true, y_pred)
    return MAE,MSE,pearson_r,R2

'''
perform the intra-sample evaluation
input: y_pred and y_true are the two single sequences
return: dtw loss (shape), tdi loss (temporal)
'''
def metrics_intra(y_pred: np.ndarray, y_true: np.ndarray):
    path, sim = dtw_path(y_pred, y_true)
    Dist = 0.0
    for i,j in path:
        Dist += (i-j)*(i-j)
    return sim, Dist/(target_window_size*target_window_size)


def remove_anomaly_points(X, y):
    assert len(X) == len(y), 'Features and targets do not have the same length.'
    normal_cond = (np.abs(zscore(y)) < 3).all(axis=1)
    X = X[normal_cond]
    y = y[normal_cond]
    return X, y

def create_datasets(X, y, mode='train', use_CL=False):
    assert mode in ['train', 'test'], 'Mode must be either train or test.'
    X = np.array(X)
    X_id = np.apply_along_axis(lambda x: x[0], 1, X[:, :, 0])
    X = X[:, :, 1:].astype(float)

    if not use_CL:
        if mode == 'train':
            X_mean, X_std = np.mean(X.reshape(-1,len(feature_columns)), axis=0), np.std(X.reshape(-1,len(feature_columns)), axis=0)
            y_mean, y_std = np.mean(y), np.std(y)
            np.savez('./X_y_mean_std.npz', X_mean=X_mean, X_std=X_std, y_mean=y_mean, y_std=y_std)
        elif mode == 'test':
            X_mean, X_std = np.load('./X_y_mean_std.npz')['X_mean'], np.load('./X_y_mean_std.npz')['X_std']
            y_mean, y_std = np.load('./X_y_mean_std.npz')['y_mean'], np.load('./X_y_mean_std.npz')['y_std']
        X = (X - X_mean) / X_std
        # y = (y - y_mean) / y_std
        X, y = remove_anomaly_points(X, y)

        data_pair = list(zip(X,y))
        if shuffle_dataset:
            random.shuffle(data_pair)
        data_pair = list(zip(*data_pair))
        if mode == 'test':
            return np.array(data_pair[0]), np.array(data_pair[1]), None

        X_train_all, y_train_all = np.array([np.array(data_pair[0])[:int(train_val_ratio*len(X))]]), np.array([np.array(data_pair[1])[:int(train_val_ratio*len(y))]])
        X_val_all, y_val_all = np.array([np.array(data_pair[0])[int(train_val_ratio*len(X)):]]), np.array([np.array(data_pair[1])[int(train_val_ratio*len(y)):]])
        recover_stats = [[0.0], [1.0]]
    else:
        task_set = list(set(X_id))
        task_set.sort()
        X_train_all, y_train_all, X_val_all, y_val_all = [], [], [], []
        X_test_all, y_test_all = [], []
        recover_stats = [[], []]
        task_mask = []
        task_shift_cond = False
        for _, task in enumerate(task_set):
            task_shift_cond = task_shift_cond | (X_id==task)
            if _ % task_shift_freq == task_shift_freq - 1 or _ == len(task_set) - 1:    
                X_tmp, y_tmp = X[task_shift_cond], y[task_shift_cond]
                if shuffle_dataset and mode != 'test':
                    data_pair = list(zip(X_tmp,y_tmp))
                    random.shuffle(data_pair)
                    data_pair = list(zip(*data_pair))
                    X_tmp, y_tmp = np.array(data_pair[0]), np.array(data_pair[1])

                X_tmp_mean = np.mean(X_tmp.reshape(-1,len(feature_columns)), axis=0)
                X_tmp_std = np.std(X_tmp.reshape(-1,len(feature_columns)), axis=0)
                # X_tmp = (X_tmp - X_tmp_mean) / X_tmp_std
                recover_stats[0].append(np.mean(y_tmp))
                recover_stats[1].append(np.std(y_tmp))
                # y_tmp = (y_tmp - np.mean(y_tmp)) / np.std(y_tmp)

                X_tmp, y_tmp = remove_anomaly_points(X_tmp, y_tmp)

                X_train_all.append(X_tmp[:int(train_val_ratio*len(X_tmp))])
                y_train_all.append(y_tmp[:int(train_val_ratio*len(y_tmp))])
                X_val_all.append(X_tmp[int(train_val_ratio*len(X_tmp)):])
                y_val_all.append(y_tmp[int(train_val_ratio*len(y_tmp)):])
                X_test_all.append(X_tmp)
                y_test_all.append(y_tmp)
                task_mask.extend([_]*len(X_tmp))
                task_shift_cond = False

        if mode == 'test': 
            return np.concatenate(X_test_all, axis=0), np.concatenate(y_test_all, axis=0), task_mask

    return X_train_all, y_train_all, X_val_all, y_val_all, recover_stats

def check_drifts(X_train, y_train, X_val):
    model_lstm = torch.load(model_path)
    model_lstm.fc = torch.nn.Identity()
    model_lstm.act = torch.nn.Identity()
    
    dataset_exp1 = TensorDataset(torch.from_numpy(X_train[0]).cuda(), torch.from_numpy(y_train[0]).cuda())
    train_loader_exp1 = DataLoader(dataset_exp1, batch_size=batch_size, shuffle=False)
    
    drift_detector = torchdrift.detectors.KSDriftDetector()
    torchdrift.utils.fit(train_loader_exp1, model_lstm, drift_detector, num_batches=10)
    
    feature_sample_exp2 = model_lstm(torch.from_numpy(X_train[0][12:22]).cuda())
    drift_score = drift_detector(feature_sample_exp2)
    p_val = drift_detector.compute_p_value(feature_sample_exp2)
    print(drift_score, p_val)
    exit()
