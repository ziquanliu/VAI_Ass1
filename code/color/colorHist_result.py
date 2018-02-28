import cv2
import numpy as np
import retrieve_fun as ut_fun
import pickle

for i in range(2000):
    img_name='0'*(4-len(str(i)))+str(i)
    c_img=cv2.imread('../candidate_imgs/'+img_name+'.jpg')
    hist_c=ut_fun.colorHist(c_img)
    print 'image ',i
    pickle.dump(hist_c,open('../result/color_hist/'+img_name+'_result.txt','wb'))