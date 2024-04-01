import os
import csv
import pdb
main_folder_path = "valids"
model_name = "Informer"
pred_length = 24

folder_list = [
    "CUEEData_%d_4_%s_CUEE_mv37_ftMS_btch128_sl24_ll0_pl4_dm8_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp_0"   % (pred_length, model_name),
    "CUEEData_%d_4_%s_CUEE_mv37_ftMS_btch128_sl24_ll0_pl4_dm16_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp_0"  % (pred_length, model_name),
    "CUEEData_%d_4_%s_CUEE_mv37_ftMS_btch128_sl24_ll0_pl4_dm32_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp_0"  % (pred_length, model_name),
    "CUEEData_%d_4_%s_CUEE_mv37_ftMS_btch128_sl24_ll0_pl4_dm64_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp_0"  % (pred_length, model_name),
    "CUEEData_%d_4_%s_CUEE_mv37_ftMS_btch128_sl24_ll0_pl4_dm128_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp_0" % (pred_length, model_name),
]

d_model = []
n_param = []
overall_mae_list = []
overall_mse_list = []

for folder_ in folder_list:
    setting_path_csv = os.path.join(main_folder_path, folder_, "model_setting.csv")
    result_stat_path_csv    = os.path.join(main_folder_path, folder_, "stats.csv")
    
    with open(setting_path_csv) as csv_file:
        d_reader = csv.reader(csv_file)
        d_dict   = dict(d_reader)

    with open(result_stat_path_csv) as csv_file:
        stat_reader = csv.reader(csv_file)
        stat_dict   = dict(stat_reader)

     
    d_model.append(int(d_dict["d_model"]))
    n_param.append(int(d_dict["Num-param"]))
    overall_mae_list.append(float(stat_dict["mae-overall"]))
    overall_mse_list.append(float(stat_dict["mse-overall"]))


import matplotlib.pyplot as plt

plt.close("all")  

fig, ax1 = plt.subplots(figsize=(15, 5)) 
 
ax1.plot(d_model, overall_mae_list, color='red')  
ax1.set_xlabel('#hidden dim', fontsize = 'large', color='red')
ax1.set_ylabel('MAE', fontsize = 'large')
ax1.tick_params(axis='x', colors='red')
ax1.grid(which='major', color='red', linewidth=0.8)

ax2 = ax1.twiny()
ax2.set_xlim(ax1.get_xlim())
ax2.set_xlabel('# params', fontsize='large', color='green')   
ax2.tick_params(axis='x', colors='green') 
ax2.set_xticks(d_model)
ax2.set_xticklabels(n_param)
ax2.grid(which='major', color='green', linestyle='--', linewidth=1)

 
plt.tight_layout()
plt.savefig("%s_pred_%d_validate_d-model-%d-%d.png" % (model_name, pred_length, min(d_model), max(d_model))) 
