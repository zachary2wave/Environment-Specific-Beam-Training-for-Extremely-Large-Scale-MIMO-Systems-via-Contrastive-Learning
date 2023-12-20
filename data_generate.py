import os
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from basic_parameter import *
from torch.utils.data import DataLoader
from util.util_function import CustomDataset
import scipy.io as io



def Data_generate(SNR, h_random=None):
    '''
    :return:
        channel,  dim =  [Ne, Nd]
        information, all middle variable
    '''

    SNR = 10 ** (SNR / 10)
    path = np.random.randint(low=4, high=10)
    H = 0
    S = 0
    for i in range(path):
        am = np.random.uniform()
        if am < 0.5:
            sin = np.random.uniform(-0.7, -0.5)    # the angle is located in -0.8~-0.6
        # elif 0.2 <= am < 0.4:
        #     sin = np.random.uniform(-0.5, -0.2)  # the angle is located in -0.3~-0.1
        # elif 0.4 < am < 0.6:
        #     sin = np.random.uniform(0, 0.2)      # the angle is located in 0.2~0.4
        # elif 0.6 < am < 0.7:
        #     sin = np.random.uniform(0.4, 0.5)    # the angle is located in 0.5~0.9
        else:
            sin = np.random.uniform(0.6, 0.8)    # the angle is located in 0.5~0.9
        bm = np.random.uniform()
        if bm < 0.5:
            if i == 0:
                d_los = np.random.uniform(low=0.3, high=0.45, size=[1])
                d_los = 1 / d_los
                d = d_los
                S = Steering_Vector(sin, d)
            else:
                d = np.random.uniform(low=0.1, high=0.2, size=[1])
                d = d_los + 1/d
        else:
            if i == 0:
                d_los = np.random.uniform(low=0.15, high=0.2, size=[1])
                d_los = 1 / d_los
                d = d_los
                S = Steering_Vector(sin, d)
            else:
                d = np.random.uniform(low=0.1, high=0.2, size=[1])
                d = d_los + 1/d



        Am = La / (4 * np.pi * d)
        Ph = Steering_Vector(sin, d)
        NowH = Am * Ph
        H += NowH
    ##''''''''' NLOS path''''''''''''''''



    signal = np.sqrt(np.conj(H.T).dot(H))
    real_part = np.expand_dims(1/np.sqrt(2)*np.random.normal(0, 1, size=(N, )), axis=-1)
    imag_part = np.expand_dims(1/np.sqrt(2)*np.random.normal(0, 1, size=(N, )), axis=-1)
    noise = np.concatenate((real_part, imag_part), axis=-1).view(np.complex).squeeze(-1)/np.sqrt(N)
    alpha = np.sqrt(SNR)
    Y_ = alpha * H / (1+alpha) + noise * signal / (1+alpha)
    if h_random == None:
        Y = Y_
    else:
        Y = h_random.dot(Y_)
    Y = Y * 1e3

    return Y, S, H



def generate_sample(data_size, snr):
    Y = np.zeros([data_size, N], dtype=np.complex64)
    label = np.zeros([data_size, N], dtype=np.complex64)
    H = np.zeros([data_size, N], dtype=np.complex64)
    Y_ = np.zeros([data_size, N], dtype=np.complex64)
    for i in range(data_size):
        Y[i, :], label[i, :], H[i, :] = Data_generate(snr)
    return Y, label, H, Y_

def generate_sample_random_snr(data_size):
    Y = np.zeros([data_size, N], dtype=np.complex64)
    label = np.zeros([data_size, N], dtype=np.complex64)
    H = np.zeros([data_size, N], dtype=np.complex64)
    Y_ = np.zeros([data_size, N], dtype=np.complex64)
    for i in range(data_size):
        snr = np.random.randint(0, 30)
        Y[i, :], label[i, :], H[i, :] = Data_generate(snr)
    return Y, label, H, Y_

def generate_dataset_random_snr(data_size):
    Y = np.zeros([data_size, N], dtype=np.complex64)
    label = np.zeros([data_size, N], dtype=np.complex64)
    H = np.zeros([data_size, N], dtype=np.complex64)
    Y_ = np.zeros([data_size, N], dtype=np.complex64)
    for i in range(data_size):
        snr = np.random.randint(0, 30)
        Y[i, :], label[i, :], H[i, :] = Data_generate(snr)
    data_set = CustomDataset(label, Y, H, Y_ )
    return DataLoader(data_set, batch_size=batchsize, shuffle=True)




def generate_dataset_random_snr(data_size):
    Y = np.zeros([data_size, N], dtype=np.complex64)
    label = np.zeros([data_size, N], dtype=np.complex64)
    H = np.zeros([data_size, N], dtype=np.complex64)
    Y_ = np.zeros([data_size, N], dtype=np.complex64)
    for i in range(data_size):
        snr = np.random.randint(0, 30)
        Y[i, :], label[i, :], H[i, :] = Data_generate(snr)
    data_set = CustomDataset(label, Y, H, Y_ )
    return DataLoader(data_set, batch_size=batchsize, shuffle=True)

def save_dataset_random_snr(data_size, name):
    Y = np.zeros([data_size, N], dtype=np.complex64)
    label = np.zeros([data_size, N], dtype=np.complex64)
    H = np.zeros([data_size, N], dtype=np.complex64)
    Y_ = np.zeros([data_size, N], dtype=np.complex64)
    for i in range(data_size):
        snr = np.random.randint(0, 30)
        Y[i, :], label[i, :], H[i, :] = Data_generate(snr)
        if i % batchsize == 0:
            print("now generate the", i // batchsize, "th batch data")
    np.savez(name, Y=Y, label=label, H=H, noise_power=Y_)

# def save_dataset_matlab(data_size, random_weight, gird = True ):
#     for snr in range(30):
#         Y = np.zeros([data_size, N], dtype=np.complex64)
#         label = np.zeros([data_size, N], dtype=np.complex64)
#         H = np.zeros([data_size, N], dtype=np.complex64)
#         Y_ = np.zeros([data_size, N], dtype=np.complex64)
#         for i in range(data_size):
#             Y[i, :], label[i, :], H[i, :], Y_[i, :] = Data_generate(snr, random_weight, gird)
#             if i % batchsize == 0:
#                 print("now generate the", i // batchsize, "th batch data")
#         io.savemat("Ongrid_snr%d.mat"%snr, {"dic": codebook_near_field, "Q": random_weight,
#                                      "Y": Y, "label": label, "Y_": Y_,
#                                      "label_h": H})


if __name__=="__main__":

    save_dataset_random_snr(1000 * batchsize, 'Sample_channel_V3')
    # save_dataset_random_snr(1000 * batchsize, 'Sample_offgrid_new2', random_weight = h_random, gird = False)
    # for _ in range(100):
    #     test_data1111 = generate_sample(batchsize, snr, random_weight=h_random, gird=False)

