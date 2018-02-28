import cv2
import pickle
from  sklearn.cluster import KMeans
import numpy as np

#num_sift=0
#for i in range(2000):
#    img_name = '0' * (4 - len(str(i))) + str(i)
#    sift_img=pickle.load(open('../../candidate_sift/'+img_name+'.txt','rb'))
#    num_sift+=sift_img.shape[0]

#the total number of sift feature
#X=np.zeros((num_sift,128))
#ind=0
#for i in range(2000):
#    img_name = '0' * (4 - len(str(i))) + str(i)
#    sift_img=pickle.load(open('../../candidate_sift/'+img_name+'.txt','rb'))
#    X[ind:ind+sift_img.shape[0],:]=sift_img
#    ind=ind+sift_img.shape[0]
#pickle.dump(X,open('../../candidate_sift/whole_sift.txt','wb'))
X=pickle.load(open('../../candidate_sift/whole_sift.txt','rb'))
print 'load complete'

k_result=KMeans(n_clusters=1000).fit(X)
print 'K-means complete'
pickle.dump(k_result,open('../../candidate_sift/cluster.txt','wb'))