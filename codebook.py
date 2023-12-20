import sys

import numpy as np

sys.path.append("section_BArate_and_SE/")
from data_generate_Non_strationary import *


def far_field_directionary():
    codebook = np.zeros((N, Num_grid_angle), dtype=np.complex64)
    index = 0
    for i in range(Num_grid_angle):
            sinAZ = np.sin(grid_angle[i] * np.pi / 180)
            HH = Far_field_Steering_Vector(grid_angle[i])
            codebook[:, i] = HH.T
    return codebook


def random_codebook():
    real_part = np.expand_dims(1 / np.sqrt(2) * np.random.normal(0, 1, size=(N,Num_grid)), axis=-1)
    imag_part = np.expand_dims(1 / np.sqrt(2) * np.random.normal(0, 1, size=(N,Num_grid)), axis=-1)
    codebook_random = np.concatenate((real_part, imag_part), axis=-1).view(np.complex).squeeze(-1) / np.sqrt(N)
    return codebook_random




def near_field_directionary(Codebook_grid_angle, Codebook_grid_d):
    dictionary = np.zeros((N, len(Codebook_grid_angle) * len(Codebook_grid_d)), dtype=np.complex64)
    index = 0
    for i in range(len(Codebook_grid_angle)):
        for j in range(len(Codebook_grid_d)):
            ARV = Steering_Vector(Codebook_grid_angle[i], Codebook_grid_d[j])
            dictionary[:, index] = ARV
            index += 1
    return dictionary



def codebook_search(Y, codebook_learned_first_layer, codebook_learned_after_layer):
    cal_learned = np.abs(np.matmul(Y, codebook_learned_first_layer))
    aa = np.argmax(cal_learned, axis=1)
    Y = Y[:,np.newaxis,:]
    YYY = np.abs(np.matmul(Y, codebook_learned_after_layer[aa, :, :]))
    return YYY.squeeze()




if __name__== "__main__":

    # hierarchical_construct

    codebook_near_field = near_field_directionary(grid_angle, grid_d)
    codebook_near_field = np.conj(codebook_near_field)

    codebook_far_field = far_field_directionary()
    codebook_far_field = np.conj(codebook_far_field)
    codebook_random = random_codebook()

    codebook_grid_angle = np.arange(-1, 1, 4 / N)
    deltaU = 10 / (N - 1) / (N - 1) / d_interval_x

    # from HCB_dai import codebook_first_layer, codebook_second_layer, codebook_last_layer, codebook_last_layer_reshape
    # from HCB_dai import hierarchical_search
    from HCB_xushi import chip_directionary


    from Model_single_layer import Dictionary_Learning_first_stage
    'loding the high layer'
    model = Dictionary_Learning_first_stage(N, area_number)
    checkpoint = torch.load("Log_data_save\M_1st_stage_learning-03-11-20-16\model.pkl")
    model.load_state_dict(checkpoint['net'])
    codebook_learned = model.obtain_psi()
    codebook_learned = codebook_learned.cpu().detach().numpy()
    codebook_learned = codebook_learned[0] + 1j * codebook_learned[1]
    codebook_learned_first_layer = normalizated(codebook_learned)

    'loding the low layer'

    from Model_single_layer import Dictionary_Learning_second_stage

    model = Dictionary_Learning_second_stage(N, 18)
    checkpoint = torch.load("Log_data_save\M_2nd_stage_learning-0-03-11-16-26\model.pkl")
    model.load_state_dict(checkpoint['net'])
    codebook_learned = model.obtain_psi()
    codebook_learned = codebook_learned.cpu().detach().numpy()
    codebook_learned = codebook_learned[0] + 1j * codebook_learned[1]
    codebook_learned_low_layer_0 = normalizated(codebook_learned)



    model = Dictionary_Learning_second_stage(N, 18)
    checkpoint = torch.load("Log_data_save\M_2nd_stage_learning-1-03-11-16-10\model.pkl")
    model.load_state_dict(checkpoint['net'])
    codebook_learned = model.obtain_psi()
    codebook_learned = codebook_learned.cpu().detach().numpy()
    codebook_learned = codebook_learned[0] + 1j * codebook_learned[1]
    codebook_learned_low_layer_1 = normalizated(codebook_learned)

    model = Dictionary_Learning_second_stage(N, 18)
    checkpoint = torch.load("Log_data_save/M_2nd_stage_learning-2-03-11-16-42/model.pkl")
    model.load_state_dict(checkpoint['net'])
    codebook_learned = model.obtain_psi()
    codebook_learned = codebook_learned.cpu().detach().numpy()
    codebook_learned = codebook_learned[0] + 1j * codebook_learned[1]
    codebook_learned_low_layer_2 = normalizated(codebook_learned)


    model = Dictionary_Learning_second_stage(N, 18)
    checkpoint = torch.load("Log_data_save\M_2nd_stage_learning-3-03-11-17-51\model.pkl")
    model.load_state_dict(checkpoint['net'])
    codebook_learned = model.obtain_psi()
    codebook_learned = codebook_learned.cpu().detach().numpy()
    codebook_learned = codebook_learned[0] + 1j * codebook_learned[1]
    codebook_learned_low_layer_3 = normalizated(codebook_learned)




    model = Dictionary_Learning_second_stage(N, 18)
    checkpoint = torch.load("Log_data_save\M_2nd_stage_learning-4-03-11-18-06\model.pkl")
    model.load_state_dict(checkpoint['net'])
    codebook_learned = model.obtain_psi()
    codebook_learned = codebook_learned.cpu().detach().numpy()
    codebook_learned = codebook_learned[0] + 1j * codebook_learned[1]
    codebook_learned_low_layer_4 = normalizated(codebook_learned)



    model = Dictionary_Learning_second_stage(N, 18)
    checkpoint = torch.load("Log_data_save\M_2nd_stage_learning-5-03-11-18-52\model.pkl")
    model.load_state_dict(checkpoint['net'])
    codebook_learned = model.obtain_psi()
    codebook_learned = codebook_learned.cpu().detach().numpy()
    codebook_learned = codebook_learned[0] + 1j * codebook_learned[1]
    codebook_learned_low_layer_5 = normalizated(codebook_learned)


    model = Dictionary_Learning_second_stage(N, 18)
    checkpoint = torch.load("Log_data_save\M_2nd_stage_learning-6-03-11-19-18\model.pkl")
    model.load_state_dict(checkpoint['net'])
    codebook_learned = model.obtain_psi()
    codebook_learned = codebook_learned.cpu().detach().numpy()
    codebook_learned = codebook_learned[0] + 1j * codebook_learned[1]
    codebook_learned_low_layer_6 = normalizated(codebook_learned)




    codebook_learned_low_layer = np.concatenate([codebook_learned_low_layer_0[np.newaxis, :],
                    codebook_learned_low_layer_1[np.newaxis, :],
                    codebook_learned_low_layer_2[np.newaxis, :],
                    codebook_learned_low_layer_3[np.newaxis, :],
                    codebook_learned_low_layer_4[np.newaxis, :],
                    codebook_learned_low_layer_5[np.newaxis, :],
                    codebook_learned_low_layer_6[np.newaxis, :]], axis=0)



    codebook_random = normalizated(codebook_random)
    codebook_far_field = normalizated(codebook_far_field)
    codebook_near_field = normalizated(codebook_near_field)
    chip_directionary = normalizated(chip_directionary)
    chip_directionary = np.conj(chip_directionary)

    SE_optimal = np.zeros(11)
    SE_los = np.zeros(11)
    SE_random = np.zeros(11)
    SE_far_field = np.zeros(11)
    SE_near_field = np.zeros(11)
    SE_near_field_H = np.zeros(11)
    SE_near_field_t = np.zeros(11)
    SE_chip = np.zeros(11)
    SE_learned = np.zeros(11)

    successs_max = np.zeros(11)
    successs_random = np.zeros(11)
    successs_far_field = np.zeros(11)
    successs_near_field = np.zeros(11)
    successs_near_field_H = np.zeros(11)
    successs_near_field_t = np.zeros(11)
    successs_chip = np.zeros(11)
    successs_learned = np.zeros(11)

    near_field = []
    near_field_low = []
    learned = []
    for _ in range(100):
        for flag, snr in enumerate(range(0, 31, 3)):
            print(snr)
            # snr = 30
            # Y, S, H_label, _ = generate_sample_random_snr(100)
            Y, S, H_label, _ = generate_sample(50, snr)

            norm = np.sqrt(np.abs(np.diagonal(np.matmul(H_label, H_complex(H_label)))))

            los = np.conj(normalizated(H_label.T))
            cal_prefect = np.diagonal(np.abs(np.matmul(H_label, los)))


            optimal = np.conj(normalizated(H_label.T))
            cal_optimal = np.diagonal(np.abs(np.matmul(Y, optimal)))
            SE_optimal[flag] += np.mean(cal_optimal)



            cal_random = np.abs(np.matmul(Y, codebook_random))
            cal_random = np.max(cal_random, axis=1)
            successs_random[flag] += np.sum(cal_random > 0.5 * cal_prefect)
            SE_random[flag] += np.mean(cal_random)

            cal_far_field = np.abs(np.matmul(Y, codebook_far_field))
            cal_far_field = np.max(cal_far_field, axis=1)
            successs_far_field[flag] += np.sum(cal_far_field > 0.5 * cal_prefect)
            SE_far_field[flag] += np.mean(cal_far_field)

            cal_near_field = np.abs(np.matmul(Y, codebook_near_field))
            cal_near_field_max = np.max(cal_near_field, axis=1)
            successs_near_field[flag] += np.sum(cal_near_field_max > 0.5*cal_prefect)
            SE_near_field[flag] += np.mean(cal_near_field_max)
            near_field.append(np.argmax(cal_near_field, axis=1))

            # cal_near_field_t = np.abs(np.matmul(Y, codebook_last_layer_reshape))
            # cal_near_field_t_max = np.max(cal_near_field_t, axis=1)
            # successs_near_field_t[flag] += np.sum(cal_near_field_t_max > 0.5 * cal_prefect)
            # SE_near_field_t[flag] += np.mean(cal_near_field_t_max)
            #
            #
            # cal_near_field_hierarchical = hierarchical_search(Y, codebook_first_layer, codebook_second_layer, codebook_last_layer)
            # SE_near_field_H[flag] += np.mean(cal_near_field_hierarchical)
            # successs_near_field_H[flag] += np.sum(cal_near_field_hierarchical > 0.5 * cal_prefect)

            cal_chip = np.abs(np.matmul(Y, chip_directionary))
            cal_chip_max = np.max(cal_chip, axis=1)
            successs_chip[flag] += np.sum(cal_chip_max > 0.5*cal_prefect)
            SE_chip[flag] += np.mean(cal_chip_max)
            # chip.append(np.argmax(cal_chip, axis=1))



            cal_learned = codebook_search(Y, codebook_learned_first_layer, codebook_learned_low_layer)
            # cal_learned = np.abs(np.matmul(Y, codebook_learned))
            cal_learned_max = np.max(cal_learned, axis=1)
            successs_learned[flag] += np.sum(cal_learned_max > 0.5 * cal_prefect)
            SE_learned[flag] += np.mean(cal_learned_max)
            learned.append(np.argmax(cal_learned, axis=1))

            pass

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(list(range(0, 31, 3)), SE_optimal, label='optimal', marker='+', linewidth=3)
    # plt.plot(list(range(0, 31, 3)), SE_random, label='random', marker='*', linewidth=3)
    plt.plot(list(range(0, 31, 3)), SE_far_field, label='Far-field', marker='o', linewidth=3)
    plt.plot(list(range(0, 31, 3)), SE_near_field, label='Distance-based BT', marker='^', linewidth=3)
    plt.plot(list(range(0, 31, 3)), SE_chip, label='Chirp-based BT', marker='d', linewidth=3)
    plt.plot(list(range(0, 31, 3)), SE_learned, label='Learned BT', marker='*', linewidth=3)
    plt.xlabel('SNR(dB)', fontsize=18)
    plt.ylabel('Spectral Efficiency (bit/s/Hz)', fontsize=18)
    plt.legend(loc='lower left', bbox_to_anchor=(1.05, 0.0), borderaxespad=0., fontsize=18)
    plt.savefig("SE.eps", format="eps", dpi=300, bbox_inches='tight')
    plt.savefig("SE.pdf", format="pdf", dpi=300, bbox_inches='tight')
    plt.show()


    plt.figure()
    # plt.plot(list(range(0, 31, 3)), successs_max / 400, label='optimal', marker='*', linewidth=3)
    # plt.plot(list(range(0, 31, 3)), successs_random / 5000, label='random', marker='*', linewidth=3)
    plt.plot(list(range(0, 31, 3)), successs_far_field / 5000, label='Far-field', marker='o', linewidth=3)
    plt.plot(list(range(0, 31, 3)), successs_near_field / 5000, label='Distance-based BT', marker='^', linewidth=3)
    # plt.plot(list(range(0, 31, 3)), successs_near_field_H / 1000, label='Hierarchical search', marker='^', linewidth=3)
    # plt.plot(list(range(0, 31, 3)), successs_near_field_t / 1000, label='Hierarchical search', marker='^', linewidth=3)
    plt.plot(list(range(0, 31, 3)), successs_chip / 5000, label='Chirp-based BT', marker='d', linewidth=3)
    plt.plot(list(range(0, 31, 3)), successs_learned / 5000, label='Learned BT', marker='*', linewidth=3)
    plt.xlabel('SNR(dB)', fontsize=18)
    plt.ylabel('Success Rate', fontsize=18)
    plt.legend(loc='lower left', bbox_to_anchor=(1.05, 0.0), borderaxespad=0., fontsize=18)
    plt.savefig("success_rate.eps", format="eps", dpi=300, bbox_inches='tight')
    plt.savefig("success_rate.pdf", format="pdf", dpi=300, bbox_inches='tight')


    near_field = np.concatenate(near_field)
    number_times_near_field = np.zeros(Num_grid)
    for i in range(Num_grid):
        number_times_near_field[i] = np.sum(near_field == i)

    # near_field_low = np.concatenate(near_field_low)
    # number_times_near_field_low = np.zeros(codebook_near_field_low.shape[1])
    # for i in range(codebook_near_field_low.shape[1]):
    #     number_times_near_field_low[i] = np.sum(near_field_low == i)

    learned = np.concatenate(learned)
    number_times_learned = np.zeros(codebook_learned.shape[1])
    for i in range(codebook_learned.shape[1]):
        number_times_learned[i] = np.sum(learned == i)

    plt.figure()
    plt.bar(range(Num_grid), number_times_near_field)

    plt.figure()
    plt.bar(range(codebook_learned.shape[1]), number_times_learned)
    plt.show()

    print(np.sum(number_times_learned == 0))
