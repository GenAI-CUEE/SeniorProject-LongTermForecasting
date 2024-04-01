from data_provider.dataloader_CUEE import DatasetCUEE
from torch.utils.data import Dataset, DataLoader
import os
import pdb


CUEE_ROOT = os.path.join(os.getcwd(),'dataset/CUEE')
CUEE_DATA = 'updated_measurement_Iclr_new.csv'

if __name__ == "__main__":
 
    dataset    = DatasetCUEE(root_path = CUEE_ROOT, flag='train', size=None, features='M', data_path=CUEE_DATA, target='I', scale=True, timeenc=0, freq='h', train_only=False)
    dataloader = DataLoader(dataset)
    for data_ in dataloader:
        seq_x, seq_y, seq_x_mark, seq_y_mark  = data_ 
        pdb.set_trace()
 