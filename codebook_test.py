from codebook import *


model = Dictionary_Learning(N, N)
checkpoint = torch.load("/home/zhangxy/codebookadaptive/Log_data_save/dictionary_learning02-07-14-25/model.pkl")
model.load_state_dict(checkpoint['net'])
codebook_learned = model.obtain_psi()
codebook_learned = codebook_learned.cpu().detach().numpy()
codebook_learned = codebook_learned[0] + 1j * codebook_learned[1]

codebook_random = normalizated(codebook_random)
codebook_far_field = normalizated(codebook_far_field)
codebook_near_field = normalizated(codebook_near_field)
codebook_near_field_low = normalizated(codebook_near_field_low)
codebook_learned = normalizated(codebook_learned)

SE_optimal = np.zeros(11)
SE_los = np.zeros(11)
SE_random = np.zeros(11)
SE_far_field = np.zeros(11)
SE_near_field = np.zeros(11)
SE_near_field_low = np.zeros(11)
SE_learned = np.zeros(11)

successs_max = np.zeros(11)
successs_random = np.zeros(11)
successs_far_field = np.zeros(11)
successs_near_field = np.zeros(11)
successs_near_field_low = np.zeros(11)
successs_learned = np.zeros(11)

near_field = []
near_field_low = []
learned = []


Y, S, H_label, _ = generate_sample(3000, 30)

norm = np.sqrt(np.abs(np.diagonal(np.matmul(H_label, H_complex(H_label)))))
cal_learned = np.abs(np.matmul(Y, codebook_learned))
maxvalue = np.max(cal_learned, axis=1)/norm
maxindices = np.argmax(cal_learned, axis=1)

number_times_learned = [0]*128
value_mean = [0]*128
for i in range(128):
    number_times_learned[i] = np.sum(maxindices == i)
    value_mean[i] = np.mean(maxvalue[maxindices == i]/5)

import matplotlib.pyplot as plt
plt.figure()
plt.bar(range(codebook_learned.shape[1]), number_times_learned)
plt.plot(range(codebook_learned.shape[1]), value_mean)
plt.show()

print(np.sum(number_times_learned == 0))
