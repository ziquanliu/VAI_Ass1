import cv2
import numpy as np



def quantize_img(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    [height, width] = img.shape
    # then equal probability quantizing the gray function to 64*64
    c_hist = np.zeros((256, 1))
    for i in range(256):
        if i == 255:
            c_hist[i] = np.sum(hist)
        else:
            c_hist[i] = np.sum(hist[0:i])
    prob_hist = c_hist / c_hist[-1]
    hist_q = np.zeros((64, 1))
    q = np.zeros((65, 1), dtype=int)
    for i in range(1, 64):
        num_com = np.ones((256, 1)) * (10 ** 6)
        for q_i in range(q[i] + 1, 256):
            num_com[q_i] = np.abs((1 - prob_hist[q[i - 1]]) / (64 - i + 1) + prob_hist[q[i - 1]] - prob_hist[q_i - 1])
        # print np.argmin(num_com)
        q[i] = np.argmin(num_com)
    q[64] = 255

    # calculate histogram in K-level
    img_quan = np.zeros((height, width), dtype=type(img))
    for i in range(height):
        for j in range(width):
            ind = 0
            for q_i in range(1, 64):
                if img[i, j] > q[q_i]:
                    ind += 1
                else:
                    break
            hist_q[ind] += 1
            img_quan[i, j] = ind
    return img_quan

def count_pair(img,d,i_m,j_m):
    [height,width]=img.shape
    padding_img=np.zeros((height+2*d,width+2*d),dtype=type(img))
    padding_img[d:d+height,d:d+width]=img
    count_0=0
    count_45=0
    count_90=0
    count_135=0
    for i in range(height):
        for j in range(width):
            if (padding_img[i+d,j+d]==i_m and padding_img[i+d,j+d+d]==j_m):
                count_0+=1
            if (padding_img[i+d,j+d]==i_m and padding_img[i+d,j+d-d]==j_m):
                count_0+=1
            if (padding_img[i+d,j+d]==i_m and padding_img[i,j+d+d]==j_m):
                count_45+=1
            if (padding_img[i+d,j+d]==i_m and padding_img[i+d+d,j]==j_m):
                count_45+=1
            if (padding_img[i+d,j+d]==i_m and padding_img[i+d+d,j+d]==j_m):
                count_90+=1
            if (padding_img[i+d,j+d]==i_m and padding_img[i,j+d]==j_m):
                count_90+=1
            if (padding_img[i+d,j+d]==i_m and padding_img[i+d+d,j+d+d]==j_m):
                count_135+=1
            if (padding_img[i+d,j+d]==i_m and padding_img[i,j]==j_m):
                count_135+=1
    return count_0,count_45,count_90,count_135






def cal_co_matrix(img,d):
    [height,width]=img.shape
    co_matrix=np.zeros((64,64,4))
    for i in range(height):
        for j in range(width):
            #0 degree
            if j+d<width:
                co_matrix[img[i,j],img[i,j+d],0]+=1
            if j-d>-1:
                co_matrix[img[i,j],img[i,j-d],0]+=1
            #45 degree
            if i-d>-1 and j+d<width:
                co_matrix[img[i,j],img[i-d,j+d],1]+=1
            if i+d<height and j-d>-1:
                co_matrix[img[i,j],img[i+d,j-d],1]+=1
            #90 degree
            if i+d<height:
                co_matrix[img[i,j],img[i+d,j],2]+=1
            if i-d>-1:
                co_matrix[img[i,j],img[i-d,j],2]+=1
            #135 degree
            if i+d<height and j+d<width:
                co_matrix[img[i,j],img[i+d,j+d],3]+=1
            if i-d>-1 and j-d>-1:
                co_matrix[img[i,j],img[i-d,j-d],3]+=1
    return co_matrix

