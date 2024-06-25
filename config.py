target_window_size = 30
feature_series_length = 30
feature_columns = {'var1': 380, 'var2': 30, 'var3': 52}
use_CL = False
use_avalanche = False
if use_avalanche: use_CL = True
task_shift_freq = 10
fisher_estimation_sample_size = 50
ewc_lamda = 1.0 # contribution decay of old task in online EWC
validate_on_recent_data = True # validation on all data or only recent data

train_val_ratio = 0.8
batch_size = 4 # 4 for CL
if use_CL:
    epochs = 50 # 50
else:
    epochs = 150
lr = 1e-3 # 1e-3
shuffle_dataset = False
use_DAIN_normalize = False
seq2seq_learning = True

mode = 'train'
batch_id = 'all'
brand_name = 'A'
model_path = './saved_models/model_lstm_continual({})_{}.pth'.format(use_CL, brand_name)

# override some parameters for specific modes
if mode == 'train': batch_id = 'all'
if mode == 'test': task_shift_freq = 1
if not use_CL: validate_on_recent_data = False
