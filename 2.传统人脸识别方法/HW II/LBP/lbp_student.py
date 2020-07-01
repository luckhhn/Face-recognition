# import the necessary packages
import numpy as np
import cv2

def original_lbp(image):
    """origianl local binary pattern"""
    rows = image.shape[0]
    cols = image.shape[1]
    lbp_image = np.zeros((rows - 2, cols - 2), np.uint8)

    # 计算每个像素点的lbp值，具体范围如上lbp_image
    for i in range(1,  rows -1):
        for j in range(1, cols -1):
            pass
            center = image[i][j]
            code = 0
            # code=code|2(按位或)  code<<=7等价于code=code<<7（左移8位）
            code |= (image[i - 1][j - 1] >= center) << 7
            code |= (image[i - 1][j] >= center) << 6
            code |= (image[i - 1][j + 1] >= center) << 5
            code |= (image[i][j + 1] >= center) << 4
            code |= (image[i + 1][j + 1] >= center) << 3
            code |= (image[i + 1][j] >= center) << 2
            code |= (image[i + 1][j - 1] >= center) << 1
            code |= (image[i][j - 1] >= center) << 0
            lbp_image[i - 1][j - 1] = code

    return lbp_image

def olbp(src):
    dst = np.zeros(src.shape, dtype=src.dtype)
    for i in range(1, src.shape[0] - 1):
        for j in range(1, src.shape[1] - 1):
            pass
            center = src[i][j]
            code = 0
            # code=code|2(按位或)  code<<=7等价于code=code<<7（左移8位）
            code |= (src[i - 1][j - 1] >= center) << 7
            code |= (src[i - 1][j] >= center) << 6
            code |= (src[i - 1][j + 1] >= center) << 5
            code |= (src[i][j + 1] >= center) << 4
            code |= (src[i + 1][j + 1] >= center) << 3
            code |= (src[i + 1][j] >= center) << 2
            code |= (src[i + 1][j - 1] >= center) << 1
            code |= (src[i][j - 1] >= center) << 0
            dst[i - 1][j - 1] = code
    return dst

if __name__ == '__main__':
    image = cv2.imread("./lms.jpg", 0)
    cv2.imshow("image", image)
    org_lbp_image = original_lbp(image)
    cv2.imshow("org_lbp_image", org_lbp_image)
    cv2.waitKey()



