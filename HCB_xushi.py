
'''
Hierarchical Codebook-based Beam Training for
Extremely Large-Scale Massive MIMO
Xu Shi, Student Member, IEEE, Jintao Wang, Senior Member, IEEE, Zhi Sun, Senior Member, IEEE, and
Jian Song, Fellow, IEEE

'''
from basic_parameter import *

maxK = 1/4/100
minK = 0
Lb = 0.5*maxK*N*N
deltaK = 0.5/N/N
deltaB = 0.1/Lb

k_step = np.arange(minK, maxK, deltaK)
b_step = np.arange(-1, 1, deltaB)



def chip_directionary(Codebook_grid_angle, Codebook_grid_d):
    dictionary = np.zeros((N, len(Codebook_grid_angle), len(Codebook_grid_d)), dtype=np.complex64)
    index = 0
    for i in range(len(Codebook_grid_angle)):
        for j in range(len(Codebook_grid_d)):
            ARV =  Chip_Steering_Vector(Codebook_grid_angle[i], Codebook_grid_d[j])
            dictionary[:, i, j] = ARV
    return dictionary





def Chip_Steering_Vector(intercept, slope):

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
    ARV = np.zeros(N, dtype=complex)
    for i in range(N):
        ARV[i] = 1.0 / np.sqrt(N) * np.exp(vi * k * (i*d_interval_x*intercept + slope * ((d_interval_x*i)**2)))
    return ARV

chip_directionary = chip_directionary(b_step, k_step)
chip_directionary = chip_directionary.reshape(N,len(k_step )*len(b_step))