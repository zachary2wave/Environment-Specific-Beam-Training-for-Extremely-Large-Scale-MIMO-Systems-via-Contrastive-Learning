import os
import sys

import torch

sys.path.append("/home/zhangxy/CE3/")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np

from basic_parameter import *
from data_generate import *
from codebook import codebook_near_field, normalizated
from util import logger
import time
K = 100
time_now = str(time.strftime("%m-%d-%H-%M", time.localtime()))
filename = str(os.path.basename(__file__))[:-3]
path = "Log_data_save/"+filename+"-"+time_now+"/"
configlist = ["stdout", "csv", 'tensorboard']
logger.configure(path, configlist)


# train_data_set = np.load("/home/zhangxy/codebookadaptive/Sample_channel_V3.npz")
train_data_set = np.load("Sample_channel_non_stationary.npz")
data_set = CustomDataset(train_data_set["label"], train_data_set["Y"], train_data_set["H"], train_data_set["noise_power"])
train_data = DataLoader(data_set, batch_size=batchsize, shuffle=True)

codebook_near_field = normalizated(codebook_near_field)


from Model_single_layer import Dictionary_Learning_first_stage

model = Dictionary_Learning_first_stage(N, area_number)
# checkpoint = torch.load("Log_data_save/M_1st_stage_learning-02-24-14-57/model.pkl")
# model.load_state_dict(checkpoint['net'])
model = model.cuda()

optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, verbose=True
                                                       , threshold=10, threshold_mode='rel', cooldown=50,
                                                       min_lr=1e-5, eps=1e-08)
bestSE = 10000
step = 0
for looptime in range(itertimes):
    for inloop, Dtrain in enumerate(train_data):
        Sendin, H_ori, label = Dtrain[0], Dtrain[2], Dtrain[3]
        Sendin, H_ori, label = Sendin.cuda(), H_ori.cuda(), label.cuda()
        label = label.real[:,:7]
        label_H_complex = complex_feature_trasnsmission(H_ori)

        x = model(H_ori)
        norm_H = torch.diagonal(torch.matmul(H_ori, torch.conj(H_ori.T)))
        norm_H = norm_H.real
        label_max = label.max(dim = -1, keepdim= True).values
        label[label == label_max] = norm_H
        x = x[0]**2 + x[1]**2
        L = nn.functional.mse_loss(x, label, reduction='sum')
        if torch.isnan(L):
            raise BaseException
        optimizer.zero_grad()
        L.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, norm_type=2)
        optimizer.step()
        step +=1

        if step % 100 == 0:
            # test_data_other = generate_dataset_random_snr(100)
            # Dtest = next(iter(test_data_other))
            with torch.no_grad():

                # Sendin_t, label_t, label_H_t = Dtest[0], Dtest[1], Dtest[2]
                # Sendin_t, label_t, label_H_t = Sendin_t.cuda(), label_t.cuda(), label_H_t.cuda()
                logger.record_tabular("epoch", looptime)
                logger.record_tabular("steps", step)
                logger.record_tabular("Loss", float(L))
                logger.dump_tabular()

                if looptime > 5 and abs(float(L)) < bestSE:
                    bestSE = abs(float(L))
                    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict()}
                    torch.save(state, path + "model.pkl")
                    print("the corr have reached", bestSE, "model has been saved")
