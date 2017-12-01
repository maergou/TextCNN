import numpy as np
from xiangliang import vocab_processer
import jieba
ccc = open('./cc_1.txt','rb').read().decode('utf-8')
ccc = ccc.split('\n')[:-1]
x_ceshi=[]
for x_fenci_1 in ccc:
        c1=jieba.cut(x_fenci_1,cut_all=False)
        dd1 = " ".join(c1)
        x_ceshi.append(dd1)
x=np.array(list(vocab_processer.fit_transform(x_ceshi)))
print(x)
