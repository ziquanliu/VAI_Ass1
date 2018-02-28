import pickle
import numpy as np
import cv2

result_1=pickle.load(open('../../result/train_result/0000_img_sort.txt','rb'))
result_2=pickle.load(open('../../result/train_result/0001_img_sort.txt','rb'))


with open('retrieval2000_result_chist.txt','w') as f:
    f.write('Q1: ')
    for i in range(len(result_1)):
        f.write(str(result_1[1999-i]) + ' ')
    f.write('\n')
    f.write('Q2: ')
    for i in range(len(result_2)):
        f.write(str(result_2[1999-i]) + ' ')
