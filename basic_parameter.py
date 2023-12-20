import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def complex_feature_trasnsmission(H):
    real = H.real.float()
    imag = H.imag.float()
    return torch.cat([real.unsqueeze(0), imag.unsqueeze(0)], dim=0)

def huber_loss(x, y):
    output = torch.where(torch.abs(x - y) > 1,  (x-y)**2, torch.abs(x-y))
    return torch.mean(output)

def H_complex(X):
    return np.conj(X.T)

def corr(X):
    return np.diagonal(np.matmul(H_complex(X), X))

def normalizated(X):
    corrx = np.sqrt(corr(X))[np.newaxis, :]
    corrx = corrx.repeat(X.shape[0], axis = 0)
    return X/corrx


def C(*arg):
    '''

    complex matrix multiply
    Args:
        *arg:

    Returns:

    '''
    if len(arg) < 2:
        raise BaseException
    else:
        real, imag = arg[0][0], arg[0][1]
        for m in arg[1:]:
            real2, imag2 = m[0], m[1]
            real_temp = torch.matmul(real, real2) - torch.matmul(imag, imag2)
            imag_temp = torch.matmul(real, imag2) + torch.matmul(imag, real2)
            real, imag = real_temp, imag_temp
    output = torch.cat([real.unsqueeze(0), imag.unsqueeze(0)], dim=0)
    return output

def C_dot(*arg):
    '''

    complex matrix multiply
    Args:
        *arg:

    Returns:

    '''
    if len(arg) < 2:
        raise BaseException
    else:
        real, imag = arg[0][0], arg[0][1]
        for m in arg[1:]:
            real2, imag2 = m[0], m[1]
            real_temp = real*real2 - imag * imag2
            imag_temp = real*imag2 + imag * real2
            real, imag = real_temp, imag_temp
    output = torch.cat([real.unsqueeze(0), imag.unsqueeze(0)], dim=0)
    return output


def Complex_Real_Dot(X, Y):
    '''
    multiply complex with real number
    Args:
        *arg:

    Returns:

    '''
    X_temp = torch.zeros_like(Y)
    X_temp[0] = Y[0].mm(X[0])
    X_temp[1] = Y[1].mm(X[0])
    return X_temp


def Steering_Vector(sin, d):

    '''
    Function:
        Generate
    Parameters:
        param nx, ny: number of antennas in x direction and y direction
        param az and el: vector of sin(theta) where theta is the azimuth angle,
                         vector of sin(theta) where theta is the Elevation angle
        param d: distance between two adjacent antennas
        param lamb: signal wave length
    Return:
         Matrix with each column an array response vector
    '''

    vi = np.array([-1j], dtype=complex)
    cos = np.sqrt(1-sin**2)
    dis = np.zeros(N)
    ARV = np.zeros(N, dtype=complex)
    for i in range(N):
        dis[i] = np.sqrt((d * sin - array_position[i, 0]) ** 2 + (d * cos - array_position[i, 1]) ** 2)
        ARV[i] = 1.0 / np.sqrt(N) * np.exp(vi * k * dis[i])[0]
    return ARV

def Far_field_Steering_Vector(sinAZ):

    '''
    Function:
        Generate
    Parameters:
        param nx, ny: number of antennas in x direction and y direction
        param az and el: vector of sin(theta) where theta is the azimuth angle,
                         vector of sin(theta) where theta is the Elevation angle
        param d: distance between two adjacent antennas
        param lamb: signal wave length
    Return:
         Matrix with each column an array response vector
    '''

    vi = np.array([-1j], dtype=complex)
    indx = np.arange(N)
    array_x = 1.0 / np.sqrt(N) * np.exp(vi * 2.0 * np.pi / La * d_interval_x * sinAZ * indx)[:, np.newaxis]
    return array_x

# 设置随机数种子
setup_seed(20)

'''
Some basic information
We assume the transmitter DMA is located at the orignal XOY planer 

'''

vi = np.array([-1j], dtype=complex)

c = 3e8                        # Speed of light 3.0*10**8
f = 2.8e10                     # Frequency 28G GHz
La = c/f                       # La: Lambda
k = 2*np.pi/La                 # Wavenumber

K_max = 26

itertimes = 100
batchsize = 1024


P_max_dbm = -13
ks = range(2, 13, 2)
Len = len(ks)

P_max_dbm_temp = -27
Pt1 = 10 ** (P_max_dbm_temp / 10) * (1e-3)

P_total = 10 ** (P_max_dbm / 10) * (1e-3)
Pt_final = P_total / Pt1

sigma_dbm = -114 # dbm
Sigma = 10 ** (sigma_dbm / 10) * 10 ** (-3)


SINR = 10*np.log10(Pt1/Sigma)

'''
the set for array anternna
'''
d_interval_x = 0.5*La # the distance between each element
Interval_x_FD = 0.5*La # the distance between each element
N = 128          # The number of elements
D = N * d_interval_x # 50cm * 50 cm
N_RF = 16


array_position = np.zeros((N,2))
for i in range(N):
    x_axis = d_interval_x*i
    array_position[i, 0] = x_axis


## The region of near-field
Alf = 1.2
Beta = 827.67
F_fraunhofer = 2*D**2/La          # Fraunhofer distance
F_Fresnel = (D**4/(8*La))**(1/4)  # Fresnel radius
print("----------------------------basic parameter-----------------------------")
print("Frequence:", f/1e9, "Ghz", " The number of anternna ", N)
print("The Fraunhofer distance", F_fraunhofer, "and Fresnel distanceR", F_Fresnel)


'''
the set for direction  
'''
'grid'
grid_angle = np.arange(-1, 1, 2/N)
Num_grid_angle = len(grid_angle)

# deltaU = 2/(N-1)/(N-1)/d_interval_x
deltaU = 0.07
grid_d = [F_fraunhofer]
for i in range(80):
    grid_d.append(1/(1/grid_d[-1]+deltaU))
    if grid_d[-1]<2:
        break
Num_grid_d = len(grid_d)
Num_grid = Num_grid_d * Num_grid_angle




'''
defined the area   
'''
area_number = 7
# incidate angle
sparce_angle = [-0.8, -0.5, -0.3, 0, 0.2, 0.4, 0.7]
sparce_distance = 1/np.random.uniform(0, 0.4, size=area_number)
sparce_start = [0, 10, 40, 30, 60, 50, 70 ]
sparce_end   = [60, 70, 90, 100, 120, 110, 128 ]
pass


vi = np.complex(real=0, imag=1)
h_analog = np.exp(vi*np.random.normal(0, 1, size=(N_RF, N))*2*np.pi)

real_part = np.expand_dims(np.random.normal(0, 1, size=(N_RF,)), axis=-1)
imag_part = np.expand_dims(np.random.normal(0, 1, size=(N_RF,)), axis=-1)
h_digital = np.concatenate((real_part, imag_part), axis=-1).view(np.complex).squeeze(-1)
h_digital = np.diag(h_digital)
h_random = np.matmul(h_digital, h_analog)






