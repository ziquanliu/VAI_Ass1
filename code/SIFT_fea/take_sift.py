import cv2
import numpy as np
import pickle

for i in range(2000):
    img_name = '0' * (4 - len(str(i))) + str(i)
    c_img = cv2.imread('../../candidate_imgs/' + img_name + '.jpg')
    gray_img=cv2.cvtColor(c_img,cv2.COLOR_BGR2GRAY)
    sift=cv2.SIFT()
    kp=sift.detect(gray_img,None)
    key_p,des=sift.compute(gray_img,kp)
    print 'image ',i
    pickle.dump(des,open('../../candidate_sift/'+img_name+'.txt','wb'))