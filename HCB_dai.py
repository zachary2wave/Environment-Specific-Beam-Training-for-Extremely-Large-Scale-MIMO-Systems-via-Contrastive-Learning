import numpy as np

from basic_parameter import *


def near_field_directionary_dai(Codebook_grid_angle, Codebook_grid_d):
    dictionary = np.zeros((N, len(Codebook_grid_angle), len(Codebook_grid_d)), dtype=np.complex64)
    index = 0
    for i in range(len(Codebook_grid_angle)):
        for j in range(len(Codebook_grid_d)):
            ARV = Steering_Vector(Codebook_grid_angle[i], Codebook_grid_d[j])
            dictionary[:, i, j] = ARV
    return dictionary



# first layer
grid_angle1 = np.arange(-1, 1, 8/N)

# deltaU = 2/(N-1)/(N-1)/d_interval_x
deltaU = 0.07*4
grid_d1 = [F_fraunhofer]
for i in range(80):
    grid_d1.append(1/(1/grid_d1[-1]+deltaU))
    if grid_d1[-1]<2:
        break
codebook_first_layer = near_field_directionary_dai(grid_angle1, grid_d1)

# second_layer
grid_angle2 = np.arange(-1, 1, 4/N)


# deltaU = 2/(N-1)/(N-1)/d_interval_x
deltaU = 0.07*2
grid_d2 = [F_fraunhofer]
for i in range(80):
    grid_d2.append(1/(1/grid_d2[-1]+deltaU))
    if grid_d2[-1]<2:
        break
codebook_second_layer = near_field_directionary_dai(grid_angle2, grid_d2)

# last_layer
grid_angle = np.arange(-1, 1, 2/N)

# deltaU = 2/(N-1)/(N-1)/d_interval_x
deltaU = 0.07
grid_d = [F_fraunhofer]
for i in range(80):
    grid_d.append(1/(1/grid_d[-1]+deltaU))
    if grid_d[-1]<2:
        break
codebook_last_layer = near_field_directionary_dai(grid_angle, grid_d)
codebook_last_layer_reshape = codebook_last_layer.reshape(N, 1024)

pass

def hierarchical_search(Y, codebook_first_layer, codebook_second_layer, codebook_last_layer):

    # first layer
    specture_efficient = []
    for i in range(50):
        first_layer_output = np.matmul(Y[i,:], codebook_first_layer.reshape(N, 3*32))
        first_layer_output = np.argmax(first_layer_output, axis=-1)
        first_layer_output_a = (first_layer_output) % 32
        first_layer_output_r = (first_layer_output) // 32

        first_layer_output_a_candidate = [np.fmax(first_layer_output_a*2 - 1, np.zeros_like(first_layer_output_a)), first_layer_output_a*2,
                                          np.fmin(first_layer_output_a*2 + 1, 63*np.ones_like(first_layer_output_a))]
        first_layer_output_r_candidate = [np.fmax(first_layer_output_r*2 - 1, np.zeros_like(first_layer_output_r)), first_layer_output_r*2,
                                          np.fmin(first_layer_output_r*2 + 1, 4*np.ones_like(first_layer_output_r))]
        select_beam = []
        for i in range(3):
            select_beam.append(codebook_second_layer[:, first_layer_output_a_candidate, first_layer_output_r_candidate[i]])
        select_beam = np.concatenate(select_beam, axis = -1)
        second_layer_output = np.matmul(Y[i,:], select_beam)
        second_layer_output = np.argmax(second_layer_output, axis=-1)
        second_layer_output_a = first_layer_output_a_candidate[(second_layer_output) % 3]
        second_layer_output_r = first_layer_output_r_candidate[(second_layer_output) // 3]


        last_layer_output_a_candidate = [np.fmax(second_layer_output_a * 2 - 1, np.zeros_like(second_layer_output_a)),
                                          second_layer_output_a * 2,
                                          np.fmin(second_layer_output_a * 2 + 1,
                                                  127 * np.ones_like(second_layer_output_a))]
        last_layer_output_r_candidate = [np.fmax(second_layer_output_r * 2 - 1, np.zeros_like(second_layer_output_r)),
                                          np.fmin(second_layer_output_r * 2, 7 * np.ones_like(second_layer_output_r)),
                                          np.fmin(second_layer_output_r * 2 + 1, 7 * np.ones_like(second_layer_output_r))]
        select_beam = []
        for i in range(3):
            select_beam.append(codebook_last_layer[:, last_layer_output_a_candidate, last_layer_output_r_candidate[i]])
        select_beam = np.concatenate(select_beam, axis = -1)
        output = np.abs(np.matmul(Y[i,:], select_beam))
        output = np.max(output, axis=-1)
        specture_efficient.append(output)
    return np.array(specture_efficient)



def hierarchical_search_a(Y, codebook_first_layer, codebook_second_layer, codebook_last_layer):

    # first layer
    specture_efficient = []
    for i in range(50):
        first_layer_output = np.matmul(Y[i,:], codebook_first_layer.reshape(N, 3*32))
        first_layer_output = np.argmax(first_layer_output, axis=-1)
        first_layer_output_a = (first_layer_output) % 32
        first_layer_output_r = (first_layer_output) // 32

        first_layer_output_a_candidate = [np.fmax(first_layer_output_a*2 - 1, np.zeros_like(first_layer_output_a)), first_layer_output_a*2,
                                          np.fmin(first_layer_output_a*2 + 1, 63*np.ones_like(first_layer_output_a))]
        first_layer_output_r_candidate = [np.fmax(first_layer_output_r*2 - 1, np.zeros_like(first_layer_output_r)), first_layer_output_r*2,
                                          np.fmin(first_layer_output_r*2 + 1, 4*np.ones_like(first_layer_output_r))]
        select_beam = []
        for i in range(3):
            select_beam.append(codebook_second_layer[:, first_layer_output_a_candidate, first_layer_output_r_candidate[i]])
        select_beam = np.concatenate(select_beam, axis = -1)
        second_layer_output = np.matmul(Y[i,:], select_beam)
        second_layer_output = np.argmax(second_layer_output, axis=-1)
        second_layer_output_a = first_layer_output_a_candidate[(second_layer_output) % 3]
        second_layer_output_r = first_layer_output_r_candidate[(second_layer_output) // 3]


        last_layer_output_a_candidate = [np.fmax(second_layer_output_a * 2 - 1, np.zeros_like(second_layer_output_a)),
                                          second_layer_output_a * 2,
                                          np.fmin(second_layer_output_a * 2 + 1,
                                                  127 * np.ones_like(second_layer_output_a))]
        last_layer_output_r_candidate = [np.fmax(second_layer_output_r * 2 - 1, np.zeros_like(second_layer_output_r)),
                                          np.fmin(second_layer_output_r * 2, 7 * np.ones_like(second_layer_output_r)),
                                          np.fmin(second_layer_output_r * 2 + 1, 7 * np.ones_like(second_layer_output_r))]
        select_beam = []
        for i in range(3):
            select_beam.append(codebook_last_layer[:, last_layer_output_a_candidate, last_layer_output_r_candidate[i]])
        select_beam = np.concatenate(select_beam, axis = -1)
        output = np.matmul(Y[i,:], select_beam)
        output = np.abs(np.max(output, axis=-1))
        specture_efficient.append(output)
    return np.array(specture_efficient)