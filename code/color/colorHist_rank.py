import cv2
import numpy as np
import retrieve_fun as ut_fun
import pickle
import operator

score=pickle.load(open('../../result/train_result/0000_score.txt','rb'))
dic_score={}
for i in range(2000):
    dic_score[str(i)]=score[i]

sort_score=sorted(dic_score.items(),key=operator.itemgetter(1))
sort_img=[]
for i in range(2000):
    sort_img.append(sort_score[i].__getitem__(0))
    print sort_score[i]
pickle.dump(sort_img,open('../../result/train_result/0000_img_sort.txt','wb'))


