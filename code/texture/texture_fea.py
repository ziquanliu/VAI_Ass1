import cv2
import numpy as np
import texture_fun as t_fun
import pickle
#d=1
#test_m=np.array([[0,0,1,1],[0,0,1,1],[0,2,2,2],[2,2,3,3]])

#print test_m
#result=t_fun.cal_co_matrix(test_m,1)
#print result[:,:,0]
#print result[:,:,1]
#print result[:,:,2]
#print result[:,:,3]

for i in range(2000):
    img_name = '0' * (4 - len(str(i))) + str(i)
    c_img = cv2.imread('../../candidate_imgs/' + img_name + '.jpg',0)
    quan_img = t_fun.quantize_img(c_img)
    co_occ = t_fun.cal_co_matrix(quan_img, 2)
    pickle.dump(co_occ,open('../../candidate_coocc_m/'+img_name+'.txt','wb'))
    print 'img',i

