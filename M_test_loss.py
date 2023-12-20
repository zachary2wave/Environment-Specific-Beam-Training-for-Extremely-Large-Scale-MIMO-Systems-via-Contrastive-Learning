import os
import sys
sys.path.append("/home/zhangxy/CE3/")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np

from basic_parameter import *
from data_generate_Non_strationary import *
from util.util_function import sparse_degree
from util import logger
import time
from codebook import *
K = 100
time_now = str(time.strftime("%m-%d-%H-%M", time.localtime()))
path = "Log_data_save/dictionary_learning"+time_now+"/"
configlist = ["stdout", "csv", 'tensorboard']
logger.configure(path, configlist)


train_data_set = np.load("Sample_channel_non_stationary.npz")
data_set = CustomDataset(train_data_set["label"], train_data_set["Y"], train_data_set["H"], train_data_set["noise_power"])
train_data = DataLoader(data_set, batch_size=batchsize, shuffle=True)



from neural_network_for_compare import Dictionary_Learning
model = Dictionary_Learning(N, Num_grid, codebook_near_field)
# checkpoint = torch.load("Log_data_save/dictionary_learning12291600/model.pkl")
# model.load_state_dict(checkpoint['net'])
model = model.cuda()

optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=100, verbose=True
                                                       , threshold=10, threshold_mode='rel', cooldown=50,
                                                       min_lr=1e-6, eps=1e-08)
bestL = 1e5
for looptime in range(itertimes):
    for inloop, Dtrain in enumerate(train_data):
        Sendin, label, label_H = Dtrain[0], Dtrain[1], Dtrain[2]
        Sendin, label, label_H = Sendin.cuda(), label.cuda(), label_H.cuda()
        label_H.numpy


