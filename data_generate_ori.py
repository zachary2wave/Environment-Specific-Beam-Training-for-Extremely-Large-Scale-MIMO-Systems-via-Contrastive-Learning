import os
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from basic_parameter import *
from torch.utils.data import DataLoader
from util.util_function import CustomDataset
import scipy.io as io



def directionary():

    dictionary = np.zeros((N, Num_grid_angle * Num_grid_d), dtype=np.complex64)
    sensing_matrix = np.zeros((N, Num_grid_angle * Num_grid_d), dtype=np.complex64)
    record_area = np.zeros((Num_grid_angle * Num_grid_d, 2))
    index = 0
    for i in range(Num_grid_angle):
        for j in range(Num_grid_d):
            ARV = Steering_Vector(grid_angle[i], grid_d[j])
            dictionary[:, index] = ARV
            record_area[index, 0] = grid_d[j]
            record_area[index, 1] = grid_angle[i]
            index += 1
    sensing_matrix = np.matmul(h_random, dictionary)
    return dictionary, sensing_matrix, record_area

codebook_near_field, sensing_matrix, record_area = directionary()
codebook_near_field = np.conj(codebook_near_field)


# G = DMA_H_matrix(Nd, Ne)
# dic_H = np.conj(codebook_near_field.T)
# real = torch.from_numpy(codebook_near_field.real).float()
# imag = torch.from_numpy(codebook_near_field.imag).float()
# dic_tensor = torch.complex(real, imag)
# dic_tensor = dic_tensor.cuda()
#
#
# dpinv = np.linalg.pinv(codebook_near_field)



real = torch.from_numpy(sensing_matrix.real).float()
imag = torch.from_numpy(sensing_matrix.imag).float()
sensing_matrix_tensor = torch.complex(real, imag)
sensing_matrix_tensor = sensing_matrix_tensor.cuda()


Num_grid = codebook_near_field.shape[1]


def Data_generate(SNR, h_random=None):
    '''
    :return:
        channel,  dim =  [Ne, Nd]
        information, all middle variable
    '''

    SNR = 10 ** (SNR / 10)
    path = np.random.randint(low=2, high=6)
    H = 0
    HH1, HH2 = 0, 0
    dis = np.zeros((path))
    grid_num = np.zeros((path),dtype=np.int32)
    angle = np.zeros((path, 2))
    F = np.zeros((path))
    Am, Ph = np.zeros((path)), np.zeros((path, N), dtype=np.complex64)
    S = np.zeros([Num_grid])
    for i in range(0, path):
        am = np.random.uniform()
        if am < 0.2:
            sin = am-0.8  # the angle is located in -0.8~-0.6
        elif 0.2 <= am < 0.4:
            sin = am-0.5  # the angle is located in -0.3~-0.1
        elif 0.4 < am < 0.6:
            sin = am-0.2    # the angle is located in 0.2~0.4
        else:
            sin = am-0.1    # the angle is located in 0.5~0.9

        d = np.random.uniform(low=0.01, high=0.45, size=[1])
        d = 1/d

        Am[i] = La / (4 * np.pi * d)
        Ph = Steering_Vector(sin, d)
        H += Am[i] * Ph

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
    S = S * 1e3
    return Y, S, H, Y_



def generate_sample(data_size, snr):
    Y = np.zeros([data_size, N], dtype=np.complex64)
    label = np.zeros([data_size, Num_grid], dtype=np.complex64)
    H = np.zeros([data_size, N], dtype=np.complex64)
    Y_ = np.zeros([data_size, N], dtype=np.complex64)
    for i in range(data_size):
        Y[i, :], label[i, :], H[i, :], Y_ [i, :] = Data_generate(snr)
    return Y, label, H, Y_



def generate_dataset_random_snr(data_size):
    Y = np.zeros([data_size, N], dtype=np.complex64)
    label = np.zeros([data_size, Num_grid], dtype=np.complex64)
    H = np.zeros([data_size, N], dtype=np.complex64)
    Y_ = np.zeros([data_size, N], dtype=np.complex64)
    for i in range(data_size):
        snr = np.random.randint(0, 30)
        Y[i, :], label[i, :], H[i, :], Y_ [i, :] = Data_generate(snr)
    data_set = CustomDataset(label, Y, H, Y_ )
    return DataLoader(data_set, batch_size=batchsize, shuffle=True)




def generate_dataset_random_snr(data_size):
    Y = np.zeros([data_size, N], dtype=np.complex64)
    label = np.zeros([data_size, Num_grid], dtype=np.complex64)
    H = np.zeros([data_size, N], dtype=np.complex64)
    Y_ = np.zeros([data_size, N], dtype=np.complex64)
    for i in range(data_size):
        snr = np.random.randint(0, 30)
        Y[i, :], label[i, :], H[i, :], Y_ [i, :] = Data_generate(snr)
    data_set = CustomDataset(label, Y, H, Y_ )
    return DataLoader(data_set, batch_size=batchsize, shuffle=True)

def save_dataset_random_snr(data_size, name):
    Y = np.zeros([data_size, N], dtype=np.complex64)
    label = np.zeros([data_size, Num_grid], dtype=np.complex64)
    H = np.zeros([data_size, N], dtype=np.complex64)
    Y_ = np.zeros([data_size, N], dtype=np.complex64)
    for i in range(data_size):
        snr = np.random.randint(0, 30)
        Y[i, :], label[i, :], H[i, :], Y_[i, :] = Data_generate(snr)
        if i % batchsize == 0:
            print("now generate the", i // batchsize, "th batch data")
    np.savez(name, Y=Y, label=label, H=H, noise_power=Y_)

def save_dataset_matlab(data_size, random_weight, gird = True ):
    for snr in range(30):
        Y = np.zeros([data_size, N], dtype=np.complex64)
        label = np.zeros([data_size, Num_grid], dtype=np.complex64)
        H = np.zeros([data_size, N], dtype=np.complex64)
        Y_ = np.zeros([data_size, N], dtype=np.complex64)
        for i in range(data_size):
            Y[i, :], label[i, :], H[i, :], Y_[i, :] = Data_generate(snr, random_weight, gird)
            if i % batchsize == 0:
                print("now generate the", i // batchsize, "th batch data")
        io.savemat("Ongrid_snr%d.mat"%snr, {"dic": codebook_near_field, "Q": random_weight,
                                     "Y": Y, "label": label, "Y_": Y_,
                                     "label_h": H})


if __name__=="__main__":

    save_dataset_random_snr(2000 * batchsize, 'Sample_channel')
    # save_dataset_random_snr(1000 * batchsize, 'Sample_offgrid_new2', random_weight = h_random, gird = False)
    # for _ in range(100):
    #     test_data1111 = generate_sample(batchsize, snr, random_weight=h_random, gird=False)

