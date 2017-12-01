
from tensorflow.contrib import learn
import numpy as np
import jieba
max_document_length = 4
cc = open('./cc_0.txt','rb').read().decode('utf-8')
cc = cc.split('\n')[:-1]
x_final=[]
for x_fenci in cc:
        c=jieba.cut(x_fenci,cut_all=False)
        dd = " ".join(c)
        x_final.append(dd)


vocab_processer=learn.preprocessing.VocabularyProcessor(20)
vocab_processer.fit(x_final)
#x = np.array(list(vocab_processer.fit_transform(x_final)))
#ccc = open('./cc_1.txt','rb').read().decode('utf-8')
#ccc = ccc.split('\n')[:-1]
#x_ceshi=[]
#for x_fenci_1 in ccc:
#        c1=jieba.cut(x_fenci_1,cut_all=False)
#        dd1 = " ".join(c1)
#        x_ceshi.append(dd1)
