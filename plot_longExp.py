import argparse
import os
import torch
from exp.infer_main import Infer_Main
import random
import numpy as np
import matplotlib.pyplot as plt
import pdb 

from datetime import datetime, timedelta

def datetime2str(time_stamp, format="%d-%H:%M"):
    pdb.set_trace()
    string_timestamp =(datetime(*np.array(time_stamp)) +  timedelta(hours=7)).strftime(format)
    return string_timestamp

def time2string(time_stamp_list, pred_len, batch_size):

    time_predictions = []

    for time_stamp_batch in time_stamp_list:

        time_predictions_array = []
        for time_stamp_ in time_stamp_batch.tolist(): 
            
            for pred_len_ind in range(pred_len):
                time_predictions_array.append(datetime2str(time_stamp_[pred_len_ind]) ) 
    
        time_predictions.append(np.array(time_predictions_array).reshape([batch_size, pred_len]))

    return time_predictions


fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')


# basic config
parser.add_argument('--mode', type=str, required=True, default="test", help='choose between {"test", "valid"}')
parser.add_argument('--train_only', type=bool, required=False, default=False, help='perform training on full input dataset without validation and testing')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, Informer, Transformer]')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

# DLinear
parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')

# PatchTST
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
#parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

# Formers 
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in',     type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
parser.add_argument('--dec_in',     type=int, default=7, help='decoder input size')
parser.add_argument('--c_out',      type=int, default=7, help='output size')
parser.add_argument('--d_model',    type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads',    type=int, default=8, help='num of heads')
parser.add_argument('--e_layers',   type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers',   type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff',       type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')


# optimization
parser.add_argument('--num_workers',   type=int, default=10, help='data loader num workers')
parser.add_argument('--itr',           type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs',  type=int, default=10, help='train epochs')
parser.add_argument('--batch_size',    type=int, default=128, help='batch size of train input data')
parser.add_argument('--patience',      type=int, default=100, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des',        type=str, default='test', help='exp description')
parser.add_argument('--loss',       type=str, default='mse', help='loss function')
parser.add_argument('--lradj',      type=str, default='type3', help='adjust learning rate')
parser.add_argument('--pct_start',  type=float, default=0.3, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)
 

ii = 0
if args.model == "PatchTST":
    setting = '{}_{}_{}_mv{}_ft{}_btch{}_sl{}_ll{}_pl{}_ps{}_st{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
        args.model_id,
        args.model,
        args.data,
        args.moving_avg,
        args.features,
        args.batch_size,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.patch_len,
        args.stride,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.distil,
        args.des, ii)
else:

    setting = '{}_{}_{}_mv{}_ft{}_btch{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
        args.model,
        args.data,
        args.moving_avg,
        args.features,
        args.batch_size,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.distil,
        args.des, ii) 


exp = Infer_Main(args)  # set experiments 
 
print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
preds, trues, time_predictions,  mae, mse, test_data = exp.run(setting, mode=args.mode) 


if not(args.embed == "timeF"):
    time_predictions = time2string(time_predictions, args.pred_len, args.batch_size)
    time_predictions = np.concatenate(time_predictions, axis=0)  

refvec_plot = [] 
seq_len = args.seq_len 
pred_len = args.pred_len 

num_pred = 500 
start = random.randint(0, preds.shape[0])
stop = start + num_pred 

fig, ax = plt.subplots(4,1, figsize=(20,15), sharey=True)  
horizon_list = [15, 30, 45, 60] 

if args.mode == "test":
    main_folder_path = "results" 
elif  args.mode == "val": 
    main_folder_path = "valids"

folder_path = os.path.join(main_folder_path, setting, 'pred-%d.png' % pred_len)
               
for pred_index in range(pred_len):

    if not(args.embed == "timeF"):
        time_predictions_ = time_predictions[start:stop,pred_index].reshape([-1])
    groundtruth_      = trues[start:stop,pred_index].reshape([-1])
    predictions_      = preds[start:stop,pred_index].reshape([-1]) 
 
    time_x = np.arange(len(groundtruth_))  
    time_x_tick = np.arange(0, len(groundtruth_), 36)

    ax[pred_index].plot(time_x, groundtruth_, label='actual', color="black")
    ax[pred_index].plot(time_x, predictions_, label='LSTM', color="red")
    ax[pred_index].set_xticks(time_x_tick)
    if not(args.embed == "timeF"):
        ax[pred_index].set_xticklabels(time_predictions_[time_x_tick] , rotation=45, ha='right') 
    else:
        ax[pred_index].set_xticklabels(time_x_tick/36, rotation=45, ha='right') 
    ax[pred_index].set_title("MAE %0.2f @ Horrizon %d mins ahead" % (mae[pred_index], horizon_list[pred_index]))
    ax[pred_index].legend()  
    ax[pred_index].grid(True)
    ax[pred_index].set_ylabel("$Watt/m^2$")

plt.tight_layout()   
plt.savefig(folder_path)
            