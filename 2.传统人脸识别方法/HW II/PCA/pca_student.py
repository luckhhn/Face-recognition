import cv2 as cv
import numpy as np


# train data
def pca_compress(data_mat, k=9999999):
    '''
    :param data_mat: 输入数据
    :param k: 前K个特征向量
    :return:  low_dim_data的数据
    '''
    # 1. 数据中心化
    data_mat = np.array(data_mat)
    data_mat = data_mat.T
    X = np.array(data_mat.mean(axis=1))
    mean_vals = X.reshape(-1, 1)  # -1表示任意行数，1表示1列
    data_mat_mean = data_mat - mean_vals

    # 2. 计算协方差矩阵
    C = (1 / data_mat_mean.shape[0]) * (data_mat_mean.dot(data_mat_mean.T)) # (10304, 10304)

    # 3. 计算特征值和特征向量
    C_matrix = np.matrix(C)
    print("------------------下面计算原始矩阵的特征值和特征向量-----------------------")
    U , eigenvalue, featurevector = np.linalg.svd(C_matrix)
    # print(featurevector.shape) # (10304, 10304)


    # 4. 降维为K维，则选择出最大的K个特征值对应特征向量组成矩阵
    eigValInd = np.argsort(eigenvalue)   # 选取前 topN 的特征值
    eigValInd = eigValInd[:(-k + 1):-1]  # 获得 topN 特征向量 构建成 转换基(转换基的数目就是映射后的特征维度)
    re_eig_vects = featurevector[eigValInd,:]  #(118,10304)


    # 5. 计算投影之后的数据
    low_dim_data = np.matmul(np.array(re_eig_vects), data_mat_mean)

    low_dim_data = low_dim_data.T  # (360, 118)
    mean_vals = mean_vals.reshape(1, -1)  # (1, 10304)
    re_eig_vects = re_eig_vects.T  # (10304, 118)


    return low_dim_data, mean_vals, re_eig_vects


# test data
def t_img(img, mean_vals, low_dim_data):
    mean_removed = img - mean_vals
    return mean_removed * low_dim_data.T


# compute the distance between vectors using euclidean distance 欧式距离
def compute_distance(vector1, vector2):
    return np.linalg.norm(np.array(vector1)[0] - np.array(vector2)[0])


# compute the distance between vectors using cosine distance 余弦距离
def compute_distance_(vector1, vector2):
    return np.dot(np.array(vector1)[0], np.array(vector2)[0]) / (np.linalg.norm(np.array(vector1)[0]) * (np.linalg.norm(np.array(vector2)[0])))


if __name__ == '__main__':

    # 1. use num 1- 9 image of each person to train
    data = []
    for i in range(1, 41):
        for j in range(1, 10):
            img = cv.imread('orl_faces/s' + str(i) + '/' + str(j) + '.pgm', 0) # =0 Return a grayscale image
            width, height = img.shape
            img = img.reshape((img.shape[0] * img.shape[1]))
            data.append(img)

    print(type(data))
    low_dim_data, mean_vals, re_eig_vects = pca_compress(data, 120)


    # 2. use num 10 image of each person to test
    correct = 0
    for k in range(1, 41):
        img = cv.imread('orl_faces/s' + str(k) + '/10.pgm', 0)
        img = img.reshape((img.shape[0] * img.shape[1]))
        print(img.shape)
        distance = t_img(img, mean_vals, low_dim_data)
        distance_mat = []
        for i in range(1, 41):
            for j in range(1, 10):
                distance_mat.append(compute_distance_(re_eig_vects[(i - 1) * 9 + j - 1], distance.reshape((1, -1)))) # 1行任意列
        num_ = np.argmax(distance_mat) # 求最大值的序号
        class_ = int(np.argmax(distance_mat) / 9) + 1
        if class_ == k:
            correct += 1
        print('s' + str(k) + '/10.pgm is the most similar to s' +
              str(class_) + '/' + str(num_ % 9 + 1) + '.pgm')
    print("accuracy: %lf" % (correct / 40))


