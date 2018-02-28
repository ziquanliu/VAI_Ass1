import cv2
import numpy as np
import retrieve_fun as ut_fun
import pickle

q_img=cv2.imread('../../train_imgs/0001.jpg')
hist_q=ut_fun.colorHist(q_img)
score=np.zeros(2000)
for i in range(2000):
    img_name = '0' * (4 - len(str(i))) + str(i)
    hist_c=pickle.load(open('../../result/color_hist/'+img_name+'_result.txt','rb'))
    score[i]=ut_fun.hist_match(hist_q,hist_c)
    print 'img ',i,',score',score[i]
pickle.dump(score,open('../../result/train_result/0001_score.txt','wb'))


