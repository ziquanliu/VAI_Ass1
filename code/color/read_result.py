import pickle
import numpy as np
import cv2

result=pickle.load(open('../result/query_result/0004_img_sort.txt','rb'))
for i in range(20):
    img_ind=result[1999-i]
    img_name = '0' * (4 - len(str(img_ind))) + str(img_ind)
    img_data=cv2.imread('../candidate_imgs/'+img_name+'.jpg')
    cv2.imshow('retrieve image',img_data)
    cv2.waitKey(0)
