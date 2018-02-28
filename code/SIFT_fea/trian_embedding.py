import cv2
import pickle


for i in range(2000):
    img_name = '0' * (4 - len(str(i))) + str(i)
    sift_img=pickle.load(open('../../candidate_sift/'+img_name+'.txt','rb'))
