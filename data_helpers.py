import numpy as np
import jieba
def load_data_and_labels(positive_data_file, negative_data_file, zhongxing_data_file, prediction_data_file):
        positive = open(positive_data_file, 'rb').read().decode('utf-8')
        negative = open(negative_data_file, 'rb').read().decode('utf-8')
        zhongxing = open(zhongxing_data_file, 'rb').read().decode('utf-8')
        prediction = open(prediction_data_file,'rb').read().decode('utf-8')

        positive_examples = positive.split('\n')[:-1]
        negative_examples = negative.split('\n')[:-1]
        zhongxing_examples = zhongxing.split('\n')[:-1]
        prediction_examples = prediction.split('\n')[:-1]

        x_text = positive_examples + negative_examples + zhongxing_examples + prediction_examples
        x_final=[]
#       x_text=[clean_str[sent] for sent in x_text]
        for x_fenchi in x_text:
                c = jieba.cut(x_fenchi, cut_all=False)
                cc = " ".join(c)
                x_final.append(cc)
        #print(x_final)
        positive_label = [[1,0,0] for _ in positive_examples]
        negative_label = [[0, 1, 0] for _ in negative_examples]
        zhongxing_label = [[0, 0, 1] for _ in zhongxing_examples]
        y = np.concatenate([positive_label, negative_label, zhongxing_label], 0)
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
