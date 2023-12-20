import os
import sys

import torch

sys.path.append("/home/zhangxy/CE3/")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np

from basic_parameter import *
from data_generate_Non_strationary import *
from codebook import codebook_near_field, normalizated
from util import logger
import time
learned_area = 0
time_now = str(time.strftime("%m-%d-%H-%M", time.localtime()))
filename = str(os.path.basename(__file__))[:-3]
path = "Log_data_save/"+filename+"-"+str(learned_area)+'-'+time_now+"/"
configlist = ["stdout", "csv", 'tensorboard']
logger.configure(path, configlist)

# train_data_set = np.load("/home/zhangxy/codebookadaptive/Sample_channel_V3.npz")
train_data_set = np.load("data_Non_strationary"+str(learned_area)+"_part.npz")
data_set = CustomDataset(train_data_set["label"], train_data_set["Y"], train_data_set["H"], train_data_set["noise_power"])
train_data = DataLoader(data_set, batch_size=batchsize, shuffle=True)

codebook_near_field = normalizated(codebook_near_field)


from Model_single_layer import Dictionary_Learning_second_stage

model = Dictionary_Learning_second_stage(N, 18)
model = model.cuda()

optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, verbose=True
                                                       , threshold=10, threshold_mode='rel', cooldown=50,
                                                       min_lr=1e-5, eps=1e-08)
bestSE = 0
step = 0
for looptime in range(itertimes):
    for inloop, Dtrain in enumerate(train_data):
        Sendin, H_ori, label = Dtrain[0], Dtrain[2], Dtrain[3]
        Sendin, H_ori, label = Sendin.cuda(), H_ori.cuda(), label.cuda()
        label = label.real
        label_H_complex = complex_feature_trasnsmission(H_ori)

        # x = model(H_ori)
        L = model.contract_learning(H_ori)
        if torch.isnan(L):
            raise BaseException
        optimizer.zero_grad()
        L.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, norm_type=2)
        optimizer.step()
        step +=1

        if step % 100 == 0:
            test_data_other = generate_dataset_random_snr_specific_area(100, learned_area)
            Dtest = next(iter(test_data_other))
            with torch.no_grad():


                Sendin_t, label_t, label_H_t = Dtest[0], Dtest[1], Dtest[2]
                Sendin_t, label_t, label_H_t = Sendin_t.cuda(), label_t.cuda(), label_H_t.cuda()

                codebook_psi = model.obtain_psi().detach().cpu().numpy()
                codebook_psi = codebook_psi[0]+1j*codebook_psi[1]
                codebook_psi = normalizated(codebook_psi)

                Normalized_Best = normalizated(label_H_t.detach().cpu().numpy().T).T

                SE_train = np.mean(np.max(np.abs(np.matmul(Sendin.detach().cpu().numpy(), codebook_psi)), axis=1))
                SE_test = np.mean(np.max(np.abs(np.matmul(Sendin_t.detach().cpu().numpy(), codebook_psi)), axis=1))
                SE_near_field = np.mean(np.max(np.abs(np.matmul(Sendin_t.detach().cpu().numpy(), codebook_near_field)), axis=1))
                SE_max = np.mean(np.diagonal(np.abs(np.matmul(Sendin_t.detach().cpu().numpy(), np.conj(Normalized_Best).T))))
                scheduler.step(SE_test)

                maxxxxx = np.argmax(np.abs(np.matmul(Sendin.detach().cpu().numpy(), codebook_psi)), axis=1)
                notselected = []
                for i in range(18):
                    if np.sum(maxxxxx==i) == 0:
                        notselected.append(i)

                logger.record_tabular("epoch", looptime)
                logger.record_tabular("steps", step)
                logger.record_tabular("Loss", float(L))
                # logger.record_tabular("Contrast", float(L_contract_learning))

                logger.record_tabular('emepty', len(notselected))
                logger.record_tabular("SE_train", float(SE_train))
                logger.record_tabular("SE_test", float(SE_test))
                logger.record_tabular("SE_near_field", float(SE_near_field))
                logger.record_tabular("SE_max", float(SE_max))

                logger.record_tabular("lr", optimizer.state_dict()["param_groups"][0]["lr"])
                logger.dump_tabular()

                if abs(float(SE_test)) > float(SE_near_field) and abs(float(SE_test)) > bestSE:
                    bestSE = abs(float(SE_test))
                    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict()}
                    torch.save(state, path + "model.pkl")
                    print("the corr have reached", bestSE, "model has been saved")