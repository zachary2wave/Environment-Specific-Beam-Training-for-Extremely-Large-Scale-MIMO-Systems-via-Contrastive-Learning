
import numpy as np
from basic_parameter import *
train_data_set = np.load("Sample_channel_non_stationary.npz")

am = train_data_set['label'].real
am = am[:,0]
data_size = 1000 * batchsize
data_area_size = []
for i in range(area_number):
    data_index = am==i
    data_area_size = sum(data_index)
    Y = train_data_set['Y'][data_index,:]
    H = train_data_set['H'][data_index,:]
    Y_ = train_data_set['noise_power'][data_index,:]
    np.savez('data_Non_strationary'+str(i)+'_part', Y=Y, label=np.zeros([data_size, area_number]), H=H, noise_power=Y_)











