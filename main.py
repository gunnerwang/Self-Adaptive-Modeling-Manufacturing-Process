import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import copy
from config import *
from utils import metrics_inter, metrics_intra, create_datasets, visualize, scatter_plot, prediction_plot, mse_plot, offline_plot
import pickle
import torch
import torch.nn as nn
from collections import deque
from avalanche.training import SynapticIntelligence, EWC
from avalanche.benchmarks.generators import tensors_benchmark
from avalanche.benchmarks.utils import AvalancheDataset
from model import LSTM, GRU_Seq2Seq, Transformer, FreTS
from dilate_loss.dilate_loss import dilate_loss
from scipy.stats import pearsonr
import matplotlib
import matplotlib.pyplot as plt


'''
X: list of arrays, each element is a task, and each task contains sub-sequences of features
y: list of arrays, each element is a task, and each task contains an array of outputs, each output is a scaler (many-to-one) or sub-sequence (many-to-many)
'''
def model_train_val(X_train_all, y_train_all, X_val_all, y_val_all, recover_stats):
    model_lstm = LSTM() if not seq2seq_learning else GRU_Seq2Seq()
    model_lstm = model_lstm.cuda()
    # model_transformer = Transformer(dim_model=8, num_heads=4, num_encoder_layers=1, num_decoder_layers=1, dropout_p=0.2)
    # model_frets = FreTS()
    loss_function = nn.MSELoss() if not seq2seq_learning else dilate_loss
    if not use_DAIN_normalize:
        optimizer = torch.optim.Adam(model_lstm.parameters(), lr=lr) # weight_decay=1e-5)
    else:
        optimizer = torch.optim.Adam([
            {'params': model_lstm.rnn.parameters(), 'lr': lr},
            {'params': model_lstm.fc.parameters(), 'lr': lr},
            {'params': model_lstm.dain.mean_layer.parameters(), 'lr': lr * model_lstm.dain.mean_lr},
            {'params': model_lstm.dain.scaling_layer.parameters(), 'lr': lr * model_lstm.dain.scale_lr},
            {'params': model_lstm.dain.gating_layer.parameters(), 'lr': lr * model_lstm.dain.gate_lr},
        ], lr=lr)  # weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs/2), gamma=0.5)

    my_scenario = tensors_benchmark(train_tensors=[(torch.tensor(X_train_all[i]).float().cuda(), torch.tensor(y_train_all[i]).float().cuda()) for i in range(len(X_train_all))],
                                    test_tensors=[(torch.tensor(X_val_all[i]).float().cuda(), torch.tensor(y_val_all[i]).float().cuda()) for i in range(len(X_val_all))],
                                    task_labels=[i for i in range(len(X_train_all))], complete_test_set_only=False)

    # print(sum(p.numel() for p in model_frets.parameters()))
    # exit()

    '''
    training with validation
    '''
    # cl_strategy = EWC(model_lstm, torch.optim.Adam(model_lstm.parameters(), lr=lr), loss_function, ewc_lambda=1e-2, train_mb_size=batch_size, train_epochs=epochs, eval_mb_size=batch_size, device='cuda')
    task_mask = []
    task_mask_allval = []
    y_pred_best_list = []
    y_true_best_list = []
    for task_idx, (X_train, y_train) in enumerate(zip(X_train_all, y_train_all)):
    # for task_idx, experience in enumerate(my_scenario.train_stream):
        X_train, y_train = X_train_all[task_idx], y_train_all[task_idx]

        print('Task index ', task_idx)
        # for g in optimizer.param_groups:
        #     g['lr'] = lr

        if validate_on_recent_data: task_mask = [task_idx] * len(X_val_all[task_idx])
        else: task_mask.extend([task_idx] * len(X_val_all[task_idx]))
        task_mask_allval.extend([task_idx] * len(X_val_all[task_idx]))

        r2_best = -10.0 # 0.0
        pr_best = -10.0 # 0.3
        dtw_best = 10.0
        tdi_best = 10.0
        y_pred_best, y_true_best = None, None

        if not use_avalanche:
            for i in range(epochs):
                print('Train epoch ', i)
                # print(optimizer.param_groups[0]['lr'])
                model_lstm.train()
                loss_windows = deque(maxlen=5)
                for s in range(0, len(X_train), batch_size):
                    optimizer.zero_grad()

                    features, target = X_train[s:s+batch_size], y_train[s:s+batch_size]
                    features, target = torch.tensor(features).cuda(), torch.tensor(target).cuda()

                    if not seq2seq_learning:
                        prediction = model_lstm(features.float())
                    else: 
                        prediction = model_lstm(features.float(), target.float(), teacher_forcing=True)

                    # print(prediction.size(), target.size())
                    prediction = prediction.squeeze(-1)
                    loss = loss_function(prediction.float(), target.float())
                    if use_CL: loss += model_lstm.ewc_loss()


                    # if s > 0 and loss > torch.mean(torch.stack(list(loss_windows))) + 2 * torch.std(torch.stack(list(loss_windows))):
                    #     for g in optimizer.param_groups:
                    #         g['lr'] *= 10
                    #     print('modify lr')

                    loss.backward()
                    optimizer.step()
                    # print(loss)
                    loss_windows.append(loss)

                '''
                validation
                '''
                if i % 5 == 0:
                    # print(len(np.concatenate(X_val_all[3:5], axis=0)))
                    # exit()
                    if validate_on_recent_data:
                        y_pred, y_true, met_values = model_test(X=X_val_all[task_idx], y=y_val_all[task_idx], model=model_lstm, recover_stats=recover_stats[task_idx])
                    else:
                        y_pred, y_true, met_values = model_test(X=np.concatenate(X_val_all[:task_idx+1], axis=0), y=np.concatenate(y_val_all[:task_idx+1], axis=0), model=model_lstm, recover_stats=np.concatenate(recover_stats[:task_idx+1], axis=0))
                    if not seq2seq_learning:
                        condition = (met_values['R2'] > r2_best and met_values['PCC'] > pr_best)
                    else:
                        condition = (met_values['R2'] > r2_best and met_values['PCC'] > pr_best and met_values['DTW'] < dtw_best and met_values['TDI'] < tdi_best)
                    if condition:
                        torch.save(model_lstm, model_path)
                        r2_best = met_values['R2']
                        pr_best = met_values['PCC']
                        dtw_best = met_values['DTW']
                        tdi_best = met_values['TDI']
                        y_pred_best = np.copy(y_pred)
                        y_true_best = np.copy(y_true)
                    # visualize(y_pred, y_true, task_mask=task_mask, task_idx=task_idx, epoch_idx=i, metrics=met_values)
                
                scheduler.step()
        else:
            res = cl_strategy.train(experience)
            if validate_on_recent_data:
                y_pred, y_true, met_values = model_test(X=X_val_all[task_idx], y=y_val_all[task_idx], model=copy.deepcopy(model_lstm), recover_stats=recover_stats[task_idx])
            else:
                y_pred, y_true, met_values = model_test(X=np.concatenate(X_val_all[:task_idx+1], axis=0), y=np.concatenate(y_val_all[:task_idx+1], axis=0), model=copy.deepcopy(model_lstm), recover_stats=recover_stats[:task_idx+1])
            if (met_values['R2'] > r2_best and met_values['PCC'] > pr_best):
                torch.save(model_lstm, model_path)
                r2_best = met_values['R2']
                pr_best = met_values['PCC']
            visualize(y_pred, y_true, task_mask=task_mask, task_idx=task_idx, metrics=met_values)
        
        y_pred_best_list.append(y_pred_best)
        y_true_best_list.append(y_true_best)

        '''
        at the end of current task, need to consolidate knowledge
        '''
        if not use_avalanche and use_CL:
            model_lstm.consolidate(model_lstm.estimate_fisher(
                    (X_train, y_train), fisher_estimation_sample_size, batch_size=batch_size
                ))

        print('\n\n\n')

    return r2_best, pr_best, np.concatenate(y_pred_best_list), np.concatenate(y_true_best_list), np.array(task_mask_allval)


def model_test(X, y, model, recover_stats):
    print('Test begins: \n --------------')

    with torch.no_grad():
        model = model.cuda()
        model.eval()
        predictions = []
        gts = []
        errors = deque(maxlen=3)

        if not seq2seq_learning: y = np.array(y).reshape(-1)
        else: y = np.array(y)

        for s in range(len(X)):
            features = torch.tensor(X[s:s+1]).cuda()
            if not seq2seq_learning: 
                prediction = model(features.float()).cpu().item()
                gt = y[s]
            else: 
                prediction = model(features.float(), torch.tensor(y[s:s+1]).float().cuda(), teacher_forcing=False).cpu().numpy().reshape(-1)
                prediction = prediction[1:]
                gt = y[s, 1:]

            predictions.append(prediction)
            gts.append(gt)
            errors.append(prediction - gt)
        
        predictions, gts = np.array(predictions), np.array(gts)
        predictions = (predictions * recover_stats[1] + recover_stats[0]) * 10.0
        gts = (gts * recover_stats[1] + recover_stats[0]) * 10.0

        if not seq2seq_learning:
            met_values = metrics_inter(predictions, gts)
            met_values = {'MAE': met_values[0], 'MSE': met_values[1], 'PCC': met_values[2], 'R2': met_values[3], 'DTW': 1.0, 'TDI': 1.0}
            print('R2 Score: {}\nPearson Correlation: {}\n'.format(met_values['R2'], met_values['PCC']))
        else:
            met_values = []
            for i in range(predictions.shape[1]):
                met_values_single = metrics_inter(predictions[:, i], gts[:, i])
                met_values.append(list(met_values_single))
            met_values = np.mean(np.array(met_values), axis=0)
            
            intra_met_values = []
            for i in range(predictions.shape[0]):
                intra_met_values_single = metrics_intra(predictions[i], gts[i])
                intra_met_values.append(list(intra_met_values_single))
            intra_met_values = np.mean(np.array(intra_met_values), axis=0)

            met_values = np.concatenate((met_values, intra_met_values))
            met_values = {'MAE': met_values[0], 'MSE': met_values[1], 'PCC': met_values[2], 'R2': met_values[3], 'DTW': met_values[4], 'TDI': met_values[5]}
            print('MSE: {}\nR2 Score: {}\nPearson Correlation: {}\nDTW: {}\nTDI: {}\n'.format(met_values['MSE'], met_values['R2'], met_values['PCC'], met_values['DTW'], met_values['TDI']))

        return predictions, gts, met_values



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    '''
    prepared datasets
    '''
    with open(os.path.join('./data', mode, 'X-%s.txt'%brand_name), 'rb') as fp:
        X = pickle.load(fp)
    with open(os.path.join('./data', mode, 'y-%s.txt'%brand_name), 'rb') as fp:
        y = pickle.load(fp)

    if mode == 'train':
        X_train_all, y_train_all, X_val_all, y_val_all, recover_stats = create_datasets(X, y, mode=mode, use_CL=use_CL)
        # check_drifts(X_train_all, y_train_all, X_val_all)
        r2_best, pr_best, y_pred_best, y_true_best, task_mask_allval = model_train_val(X_train_all, y_train_all, X_val_all, y_val_all, np.array(recover_stats).T)
        print('Best R2 Score (Val): {}, Best Pearson Correlation (Val): {}'.format(r2_best, pr_best))
        print(y_pred_best.shape, y_true_best.shape, task_mask_allval.shape)
    elif mode == 'test':
        X_test, y_test, task_mask = create_datasets(X, y, mode=mode, use_CL=use_CL)
        model_lstm = torch.load(model_path)
        model_lstm = model_lstm.cuda()
        y_pred, y_true, met_values = model_test(X_test, y_test, model_lstm)
        visualize(y_pred, y_true, task_mask=task_mask, metrics=met_values)












