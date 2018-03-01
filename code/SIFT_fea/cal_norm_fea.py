import pickle
import numpy as np
import scipy
from scipy.linalg import sqrtm
#k_means=pickle.load(open('../../sift_cluster_128.txt','rb'))
#codebook=k_means.cluster_centers_
#pickle.dump(codebook,open('../../codebook_128.txt','wb'))
codebook=pickle.load(open('../../codebook_128.txt','rb'))
num_code=codebook.shape[0]
sum_R=np.zeros((num_code*128,1))
sum_sift=0.0

for i in range(2000):
    img_name = '0' * (4 - len(str(i))) + str(i)
    sift_img=pickle.load(open('../../candidate_sift/'+img_name+'.txt','rb'))
    num_sift=sift_img.shape[0]
    sum_sift+=num_sift
    for n in range(num_sift):
        R = np.zeros((num_code*128,1))
        for j in range(num_code):
            temp_v=sift_img[n, :] - codebook[j, :]
            R[j*128:(j+1)*128,0]=temp_v/np.linalg.norm(temp_v)
        sum_R+=R
    print 'complete sum ',i

sum_cov_R=np.zeros((num_code*128,num_code*128))
mean_R=sum_R/sum_sift

for i in range(2000):
    img_name = '0' * (4 - len(str(i))) + str(i)
    sift_img=pickle.load(open('../../candidate_sift/'+img_name+'.txt','rb'))
    num_sift=sift_img.shape[0]
    sum_sift+=num_sift
    for n in range(num_sift):
        R = np.zeros((num_code*128,1))
        for j in range(num_code):
            temp_v=sift_img[n, :] - codebook[j, :]
            R[j*128:(j+1)*128,0]=temp_v/np.linalg.norm(temp_v)
        sum_cov_R+=(R-mean_R).dot((R-mean_R).transpose())
    print 'complete cov',i

cov_R=sum_cov_R/sum_sift
multi_m=sqrtm(scipy.linalg.inv(cov_R))
Rp_l=[]
for i in range(2000):
    img_name = '0' * (4 - len(str(i))) + str(i)
    sift_img=pickle.load(open('../../candidate_sift/'+img_name+'.txt','rb'))
    num_sift=sift_img.shape[0]
    sum_sift+=num_sift
    R_sum_i=np.zeros((num_code*128,1))
    for n in range(num_sift):
        R = np.zeros((num_code*128,1))
        for j in range(num_code):
            temp_v=sift_img[n, :] - codebook[j, :]
            R[j*128:(j+1)*128,0]=temp_v/np.linalg.norm(temp_v)
        R_sum_i+=R
    Rp_l.append(np.dot(multi_m,R_sum_i)-float(num_sift)*np.dot(multi_m,mean_R))
    print 'complete all',i



