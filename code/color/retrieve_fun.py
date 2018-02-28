import cv2
import numpy as np


def colorHist(img):
    i_b, i_g, i_r = cv2.split(img)
    h, w = i_b.shape
    opp_img = np.zeros([h, w, 3])
    for i in range(h):
        for j in range(w):
            # wb ranges from 0 to 765
            opp_img[i, j, 0] = float(i_b[i, j]) + float(i_g[i, j]) + float(i_r[i, j])
            # rg ranges from -255 to 255
            opp_img[i, j, 1] = float(i_r[i, j]) - float(i_b[i, j])
            # by ranges from -510 to 510
            opp_img[i, j, 2] = 2 * float(i_b[i, j]) - float(i_r[i, j]) - float(i_g[i, j])

    step_wb = 766 / 8
    step_rg = 510 / 16
    step_by = 1020 / 16
    int_img = np.zeros([h, w, 3], dtype=np.int8)
    # normalize opp_img
    for i in range(h):
        for j in range(w):
            # wb ranges from 0 to 765
            int_img[i, j, 0] = min(int(opp_img[i, j, 0] / step_wb), 7)
            # rg ranges from -255 to 255
            int_img[i, j, 1] = min(int((opp_img[i, j, 1] + 255) / step_rg), 15)
            # by ranges from -510 to 510
            int_img[i, j, 2] = min(int((opp_img[i, j, 2] + 510) / step_by), 15)

    hist_mat = np.zeros([8, 16, 16])
    for i in range(h):
        for j in range(w):
            hist_mat[int_img[i, j, 0], int_img[i, j, 1], int_img[i, j, 2]] += 1
    return hist_mat


def hist_match(q_Hist,c_Hist):
    int_num=0
    for i in range(8):
        for j in range(16):
            for k in range(16):
                int_num+=min(q_Hist[i,j,k],c_Hist[i,j,k])
    #print int_num
    m_score=float(int_num)/np.sum(c_Hist)
    return m_score