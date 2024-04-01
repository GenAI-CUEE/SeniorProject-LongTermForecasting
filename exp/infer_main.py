from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST, RLSTM
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np
import pdb
import csv
from utils.tools import save_settings_dict
warnings.filterwarnings('ignore')

class Infer_Main(Exp_Basic):
    def __init__(self, args):
        super(Infer_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
            'RLSTM': RLSTM
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion 
 

    def run(self, setting, mode="test"):
        test_data, test_loader = self._get_data(flag=mode)
         
        print('loading model')
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Number of parameters: %d" % num_params)

        if mode == "test":
            main_folder_path = "results" 
        elif mode == "val": 
            main_folder_path = "valids"

        folder_path = os.path.join(main_folder_path, setting)
        os.makedirs(main_folder_path, exist_ok=True)    
        os.makedirs(folder_path, exist_ok=True)  
        save_settings_dict(self.args, setting, num_params, folder=main_folder_path) 

        preds = []
        trues = []
        inputx = []
        timestamp_y = []
        if mode == "test":
            folder_path_ = './results_per_sample/' + setting + '/'
            if not os.path.exists(folder_path_):
                os.makedirs(folder_path_)

        self.model.eval()
        pbar = tqdm(test_loader)
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pbar):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model or "RLSTM" in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model or "RLSTM" in self.args.model:
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                 
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device) 
                
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                timestamp_y.append(batch_y_mark)
                
                if (mode == "test") and (i % 20 == 0) :
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path_, str(i) + '.png'))
                
                MSE_temp = np.mean((pred - true) ** 2)
                pbar.set_description("MSE %f" % MSE_temp)

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
            
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

        # result save

        mae_list = []
        mse_list = []
        
        performance_dict = {}

        for seq_i in range(self.args.pred_len): 

            preds_rev = test_data.inverse_transform(preds[:,seq_i,:])
            trues_rev = test_data.inverse_transform(trues[:,seq_i,:])
            mae, mse, rmse, mape, mspe, rse, corr = metric(preds_rev, trues_rev)
            mae_list.append(mae)
            mse_list.append(mse)


            performance_dict["mse-%d" % seq_i] = mse
            performance_dict["mae-%d" % seq_i] = mae
            performance_dict["rse-%d" % seq_i] = rse
            performance_dict["corr-%d" % seq_i] = corr 

            print('%d:  mse: %f, mae: %f' % (seq_i, mse, mae))
        
        performance_dict["mse-overall" ] = sum(mse_list)/self.args.pred_len
        performance_dict["mae-overall" ] = sum(mae_list)/self.args.pred_len

        print('OVERALL:  mse:{}, mae:{}'.format( sum(mse_list)/self.args.pred_len, sum(mae_list)/self.args.pred_len ))
        
 
        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(os.path.join(folder_path , 'pred.npy'), preds)
        
        performance_dict["Num-param"] = num_params

        with open(os.path.join(folder_path, 'stats.csv'), 'w') as csv_file:  
            writer = csv.writer(csv_file)
            for key, value in performance_dict.items():
                writer.writerow([key, value])

        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return preds, trues, timestamp_y, mae_list, mse_list, test_data

