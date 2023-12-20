import os
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from basic_parameter import *
from torch.utils.data import DataLoader
from util.util_function import CustomDataset
import scipy.io as io





def Data_generate_Non_strationary(SNR, am=None):
    '''
    :return:
        channel,  dim =  [Ne, Nd]
        information, all middle variable
    '''

    SNR = 10 ** (SNR / 10)
    path = np.random.randint(low=4, high=10)
    H = 0

    ##''''''''' los path''''''''''''''''
    if am ==None:
        am = np.random.randint(0, 7)

    sin = sparce_angle[am] + np.random.uniform(low=-0.1, high=0.1, size=[1])
    d_los = sparce_distance[am] + 1/np.random.uniform(low=0.4, high=1, size=[1])
    first_place = sparce_start[am]
    last_place = sparce_end[am]

    # Am = np.random.normal(0, 1)
    Am = La / (4 * np.pi * d_los )
    Ph = Steering_Vector(sin, d_los)
    NowH = Am * Ph
    NowH[0:first_place] = 0
    NowH[last_place:N] = 0
    H += NowH

    # for _ in  range( np.random.randint(0, 4)):
    #     Am = np.random.normal(0, 1e-3)
    #     sin = np.random.uniform(-1, 1)
    #     d_los = 1/np.random.uniform(0.01, 0.5)
    #     Ph = Steering_Vector(sin, d_los)
    #     NowH = Am * Ph
    #     first_place = np.random.randint(0, 60)
    #     last_place = first_place + np.random.randint(0, 60)
    #     NowH[0:first_place] = 0
    #     NowH[last_place:N] = 0
    #     H += NowH


    signal = np.sqrt(np.conj(H.T).dot(H))
    real_part = np.expand_dims(1/np.sqrt(2)*np.random.normal(0, 1, size=(N, )), axis=-1)
    imag_part = np.expand_dims(1/np.sqrt(2)*np.random.normal(0, 1, size=(N, )), axis=-1)
    noise = np.concatenate((real_part, imag_part), axis=-1).view(np.complex).squeeze(-1)/np.sqrt(N)
    alpha = np.sqrt(SNR)
    Y_ = alpha * H / (1+alpha) + noise * signal / (1+alpha)
    # if h_random == None:
    #     Y = Y_
    # else:
    #     Y = h_random.dot(Y_)
    Y = Y_ * 1e3
    H = H * 1e3
    index_vector = np.zeros([area_number])
    index_vector[am] = np.sum(np.abs(H))*1e3
    label = index_vector
    return Y, am, H, label



def generate_sample(data_size, snr):
    Y = np.zeros([data_size, N], dtype=np.complex64)
    label = np.zeros([data_size,1])
    H = np.zeros([data_size, N], dtype=np.complex64)
    Y_ = np.zeros([data_size, area_number], dtype=np.complex64)
    for i in range(data_size):
        Y[i, :], label[i, :], H[i, :], Y_ [i, :] = Data_generate_Non_strationary(snr)
    return Y, label, H, Y_



def generate_dataset_random_snr(data_size):
    Y = np.zeros([data_size, N], dtype=np.complex64)
    label = np.zeros([data_size, N], dtype=np.complex64)
    H = np.zeros([data_size, N], dtype=np.complex64)
    Y_ = np.zeros([data_size, area_number], dtype=np.complex64)
    for i in range(data_size):
        snr = np.random.randint(0, 30)
        Y[i, :], label[i, :], H[i, :], Y_ [i, :] = Data_generate_Non_strationary(snr)
    data_set = CustomDataset(label, Y, H, Y_ )
    return DataLoader(data_set, batch_size=batchsize, shuffle=True)



def generate_dataset_random_snr_specific_area(data_size, am):
    Y = np.zeros([data_size, N], dtype=np.complex64)
    label = np.zeros([data_size, N], dtype=np.complex64)
    H = np.zeros([data_size, N], dtype=np.complex64)
    Y_ = np.zeros([data_size, area_number], dtype=np.complex64)
    for i in range(data_size):
        snr = np.random.randint(0, 30)
        Y[i, :], label[i, :], H[i, :], Y_ [i, :] = Data_generate_Non_strationary(snr, am)
    data_set = CustomDataset(label, Y, H, Y_ )
    return DataLoader(data_set, batch_size=batchsize, shuffle=True)




def save_dataset_random_snr(data_size, name):
    Y = np.zeros([data_size, N], dtype=np.complex64)
    label = np.zeros([data_size, N], dtype=np.complex64)
    H = np.zeros([data_size, N], dtype=np.complex64)
    Y_ = np.zeros([data_size, area_number], dtype=np.complex64)
    for i in range(data_size):
        snr = np.random.randint(0, 30)
        Y[i, :], label[i, :], H[i, :], Y_[i, :] = Data_generate_Non_strationary(snr)
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
#             Y[i, :], label[i, :], H[i, :], Y_[i, :] = Data_generate_Non_strationary(snr, random_weight, gird)
#             if i % batchsize == 0:
#                 print("now generate the", i // batchsize, "th batch data")
#         io.savemat("Ongrid_snr%d.mat"%snr, {"dic": codebook_near_field, "Q": random_weight,
#                                      "Y": Y, "label": label, "Y_": Y_,
#                                      "label_h": H})


if __name__=="__main__":

    save_dataset_random_snr(1000 * batchsize, 'Sample_channel_non_stationary')
    # save_dataset_random_snr(1000 * batchsize, 'Sample_offgrid_new2', random_weight = h_random, gird = False)
    # for _ in range(100):
    #     test_data1111 = generate_sample(batchsize, snr, random_weight=h_random, gird=False)

