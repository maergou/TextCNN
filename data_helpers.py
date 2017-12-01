import numpy as np
import jieba


def load_data_and_labels(sample_0_data_file,
                         sample_1_data_file,
                         sample_2_data_file,
                         sample_3_data_file,
                         sample_4_data_file,
                         sample_5_data_file,
                         sample_6_data_file,
                         sample_7_data_file,
                         sample_8_data_file,
                         sample_9_data_file,
                         sample_10_data_file,
                         sample_11_data_file,
                         sample_12_data_file,
                         sample_13_data_file,
                         sample_14_data_file,
                         sample_15_data_file,
                         sample_16_data_file,
                         sample_17_data_file,
                         sample_18_data_file,
                         sample_19_data_file,
                         sample_20_data_file,
                         sample_21_data_file,
                         sample_22_data_file,
                         sample_23_data_file,
                         sample_24_data_file,
                         sample_25_data_file,
                         sample_26_data_file,
                         sample_27_data_file,
                         sample_28_data_file,
                         sample_29_data_file,
                         sample_30_data_file,
                         sample_31_data_file,
                         sample_32_data_file,
                         sample_33_data_file
                         ):
    sample_0 = open(sample_0_data_file, 'rb').read().decode('utf-8')
    sample_1 = open(sample_1_data_file, 'rb').read().decode('utf-8')
    sample_2 = open(sample_2_data_file, 'rb').read().decode('utf-8')
    sample_3 = open(sample_3_data_file, 'rb').read().decode('utf-8')
    sample_4 = open(sample_4_data_file, 'rb').read().decode('utf-8')
    sample_5 = open(sample_5_data_file, 'rb').read().decode('utf-8')
    sample_6 = open(sample_6_data_file, 'rb').read().decode('utf-8')
    sample_7 = open(sample_7_data_file, 'rb').read().decode('utf-8')
    sample_8 = open(sample_8_data_file, 'rb').read().decode('utf-8')
    sample_9 = open(sample_9_data_file, 'rb').read().decode('utf-8')
    sample_10 = open(sample_10_data_file, 'rb').read().decode('utf-8')
    sample_11 = open(sample_11_data_file, 'rb').read().decode('utf-8')
    sample_12 = open(sample_12_data_file, 'rb').read().decode('utf-8')
    sample_13 = open(sample_13_data_file, 'rb').read().decode('utf-8')
    sample_14 = open(sample_14_data_file, 'rb').read().decode('utf-8')
    sample_15 = open(sample_15_data_file, 'rb').read().decode('utf-8')
    sample_16 = open(sample_16_data_file, 'rb').read().decode('utf-8')
    sample_17 = open(sample_17_data_file, 'rb').read().decode('utf-8')
    sample_18 = open(sample_18_data_file, 'rb').read().decode('utf-8')
    sample_19 = open(sample_19_data_file, 'rb').read().decode('utf-8')
    sample_20 = open(sample_20_data_file, 'rb').read().decode('utf-8')
    sample_21 = open(sample_21_data_file, 'rb').read().decode('utf-8')
    sample_22 = open(sample_22_data_file, 'rb').read().decode('utf-8')
    sample_23 = open(sample_23_data_file, 'rb').read().decode('utf-8')
    sample_24 = open(sample_24_data_file, 'rb').read().decode('utf-8')
    sample_25 = open(sample_25_data_file, 'rb').read().decode('utf-8')
    sample_26 = open(sample_26_data_file, 'rb').read().decode('utf-8')
    sample_27 = open(sample_27_data_file, 'rb').read().decode('utf-8')
    sample_28 = open(sample_28_data_file, 'rb').read().decode('utf-8')
    sample_29 = open(sample_29_data_file, 'rb').read().decode('utf-8')
    sample_30 = open(sample_30_data_file, 'rb').read().decode('utf-8')
    sample_31 = open(sample_31_data_file, 'rb').read().decode('utf-8')
    sample_32 = open(sample_32_data_file, 'rb').read().decode('utf-8')
    sample_33 = open(sample_33_data_file, 'rb').read().decode('utf-8')

    sample_0_examples = sample_0.split('\n')[:-1]
    sample_1_examples = sample_1.split('\n')[:-1]
    sample_2_examples = sample_2.split('\n')[:-1]
    sample_3_examples = sample_3.split('\n')[:-1]
    sample_4_examples = sample_4.split('\n')[:-1]
    sample_5_examples = sample_6.split('\n')[:-1]
    sample_6_examples = sample_6.split('\n')[:-1]
    sample_7_examples = sample_7.split('\n')[:-1]
    sample_8_examples = sample_8.split('\n')[:-1]
    sample_9_examples = sample_9.split('\n')[:-1]
    sample_10_examples = sample_10.split('\n')[:-1]
    sample_11_examples = sample_11.split('\n')[:-1]
    sample_12_examples = sample_12.split('\n')[:-1]
    sample_13_examples = sample_13.split('\n')[:-1]
    sample_14_examples = sample_14.split('\n')[:-1]
    sample_15_examples = sample_15.split('\n')[:-1]
    sample_16_examples = sample_16.split('\n')[:-1]
    sample_17_examples = sample_17.split('\n')[:-1]
    sample_18_examples = sample_18.split('\n')[:-1]
    sample_19_examples = sample_19.split('\n')[:-1]
    sample_20_examples = sample_20.split('\n')[:-1]
    sample_21_examples = sample_21.split('\n')[:-1]
    sample_22_examples = sample_22.split('\n')[:-1]
    sample_23_examples = sample_23.split('\n')[:-1]
    sample_24_examples = sample_24.split('\n')[:-1]
    sample_25_examples = sample_25.split('\n')[:-1]
    sample_26_examples = sample_26.split('\n')[:-1]
    sample_27_examples = sample_27.split('\n')[:-1]
    sample_28_examples = sample_28.split('\n')[:-1]
    sample_29_examples = sample_29.split('\n')[:-1]
    sample_30_examples = sample_30.split('\n')[:-1]
    sample_31_examples = sample_31.split('\n')[:-1]
    sample_32_examples = sample_32.split('\n')[:-1]
    sample_33_examples = sample_33.split('\n')[:-1]
    x_text = sample_0_examples + sample_1_examples + sample_2_examples + sample_3_examples + sample_4_examples + sample_5_examples + sample_6_examples +


sample_7_examples + sample_8_examples + sample_9_examples + sample_10_examples + sample_11_examples + sample_12_examples + sample_13_examples + sample
_14_examples + sample_15_examples + sample_16_examples + sample_17_examples + sample_18_examples + sample_19_examples + sample_20_examples + sample_21_e
xamples + sample_22_examples + sample_23_examples + sample_24_examples + sample_25_examples + sample_26_examples + sample_27_examples + sample_28_exampl
es + sample_29_examples + sample_30_examples + sample_31_examples + sample_32_examples + sample_33_examples
x_final = []
#       x_text=[clean_str[sent] for sent in x_text]
for x_fenchi in x_text:
    c = jieba.cut(x_fenchi, cut_all=False)
    cc = " ".join(c)
    x_final.append(cc)
    # print(x_final)

sample_0_label = [
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in
    sample_0_examples]
sample_1_label = [
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in
    sample_1_examples]
sample_2_label = [
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in
    sample_2_examples]
sample_3_label = [
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in
    sample_3_examples]
sample_4_label = [
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in
    sample_4_examples]
sample_5_label = [
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in
    sample_5_examples]
sample_6_label = [
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in
    sample_6_examples]
sample_7_label = [
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in
    sample_7_examples]
sample_8_label = [
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in
    sample_8_examples]
sample_9_label = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in
    sample_9_examples]
sample_10_label = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in
    sample_10_examples]
sample_11_label = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in
    sample_11_examples]
sample_12_label = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in
    sample_12_examples]
sample_13_label = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in
    sample_13_examples]
sample_14_label = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in
    sample_14_examples]
sample_15_label = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in
    sample_15_examples]
sample_16_label = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in
    sample_16_examples]
sample_17_label = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in
    sample_17_examples]
sample_18_label = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in
    sample_18_examples]
sample_19_label = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in
    sample_19_examples]
sample_20_label = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in
    sample_20_examples]
sample_21_label = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in
    sample_21_examples]
sample_22_label = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in
    sample_22_examples]
sample_23_label = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in
    sample_23_examples]
sample_24_label = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in
    sample_24_examples]
sample_25_label = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0] for _ in
    sample_25_examples]
sample_26_label = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0] for _ in
    sample_26_examples]
sample_27_label = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0] for _ in
    sample_27_examples]
sample_28_label = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0] for _ in
    sample_28_examples]
sample_29_label = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0] for _ in
    sample_29_examples]
sample_30_label = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0] for _ in
    sample_30_examples]
sample_31_label = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0] for _ in
    sample_31_examples]
sample_32_label = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0] for _ in
    sample_32_examples]
sample_33_label = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] for _ in
    sample_33_examples]
y = np.concatenate([sample_0_label, sample_1_label, sample_2_label, sample_3_label, sample_4_label, sample_5_label,
                    sample_6_label, sample_7_label, sample_8_label, sample_9_label, sample_10_label, sample_11_label,
                    sample_12_label, sample_13_label, sample_14_label, sample_15_label, sample_16_label,
                    sample_17_label,
                    sample_18_label, sample_19_label, sample_20_label, sample_21_label, sample_22_label,
                    sample_23_label,
                    sample_24_label, sample_25_label, sample_26_label, sample_27_label, sample_28_label,
                    sample_29_label,
                    sample_30_label, sample_31_label, sample_32_label, sample_33_label], 0)

print(len(y))
print(len(x_final))
return [x_final, y, len(y)]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

