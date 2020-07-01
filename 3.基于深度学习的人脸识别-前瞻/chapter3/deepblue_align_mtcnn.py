import cv2
import numpy as np
from align_faces import warp_and_crop_face, get_reference_facial_points
from mtcnn.detector import MtcnnDetector

def draw_all(filename, bbox, facial5points):
    image = cv2.imread(filename)
    for i in range(bbox.shape[0]):
        x, y, r, b, _ = list(map(int, bbox[i]))
        lms = list(map(int, facial5points[i]))
        w = r - x + 1
        h = b - y + 1
        cv2.rectangle(image, (x, y), (x + w, y + h), (0,255,0), 2, 16)
        cv2.circle(image, (lms[0], lms[5]), 2, (0, 255, 0), 2)
        cv2.circle(image, (lms[1], lms[6]), 2, (0, 255, 0), 2)
        cv2.circle(image, (lms[2], lms[7]), 2, (0, 255, 0), 2)
        cv2.circle(image, (lms[3], lms[8]), 2, (0, 255, 0), 2)
        cv2.circle(image, (lms[4], lms[9]), 2, (0, 255, 0), 2)
    dst_name = filename[: filename.rfind('.')]
    cv2.imwrite("{}_all.jpg".format(dst_name), image)


def draw_bbox(filename, bbox):
    image = cv2.imread(filename)
    for i in range(bbox.shape[0]):
        x, y, r, b, _ = list(map(int, bbox[i]))
        w = r - x + 1
        h = b - y + 1
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2, 16)
    dst_name = filename[: filename.rfind('.')]
    cv2.imwrite("{}_box.jpg".format(dst_name), image)

def draw_lms(filename, facial5points):
    image = cv2.imread(filename)
    for i in range(facial5points.shape[0]):
        lms = list(map(int, facial5points[i]))  # 会根据提供的函数(int)对指定序列(facial5points)做映射
        # landms
        cv2.circle(image, (lms[0], lms[5]), 2, (0, 255, 0), 2)
        cv2.circle(image, (lms[1], lms[6]), 2, (0, 255, 0), 2)
        cv2.circle(image, (lms[2], lms[7]), 2, (0, 255, 0), 2)
        cv2.circle(image, (lms[3], lms[8]), 2, (0, 255, 0), 2)
        cv2.circle(image, (lms[4], lms[9]), 2, (0, 255, 0), 2)
    dst_name = filename[: filename.rfind('.')]
    cv2.imwrite("{}_lms.jpg".format(dst_name), image)

def draw_name(filename, bbox, labels, thickness=2):
    image = cv2.imread(filename)
    x, y, r, b, _ = list(map(int, bbox))
    w = r - x + 1
    h = b - y + 1
    draw = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), thickness, 16)
    pos = (x + 3, y - 5)
    cv2.putText(image, str(labels), pos, 0, 0.5, (0, 255, 0), 2, 16)
    dst_name = filename[: filename.rfind('.')]
    cv2.imwrite("{}_name.jpg".format(dst_name), draw)


def process(filename, type, output_size):
    img = cv2.imread(filename)
    bbox, facial5points = detector.detect_faces(img)

    # # visualization
    draw_bbox(filename, bbox)
    draw_lms(filename, facial5points)
    draw_all(filename, bbox, facial5points)

    # # show ref_5pts
    if output_size[0] == 112 and output_size[1] == 112:
        tmp_pts = np.array([[38.29459953,
                             73.53179932,
                             56.02519989,
                             41.54930115,
                             70.72990036,
                             51.69630051,
                             51.50139999,
                             71.73660278,
                             92.3655014,
                             92.20410156]])
        empty_face = np.zeros((112, 112, 3))
        cv2.imwrite('empty_face.jpg', empty_face)
        draw_lms('empty_face.jpg', tmp_pts)
    elif output_size[0] == 96 and output_size[1] == 112:
        tmp_pts = np.array([[30.29459953,
                             65.53179932,
                             48.02519989,
                             33.54930115,
                             62.72990036,
                             51.69630051,
                             51.50139999,
                             71.73660278,
                             92.3655014,
                             92.20410156]])
        empty_face = np.zeros((96, 112, 3))
        cv2.imwrite('empty_face_'+str(output_size[0])
                    + '_' + str(output_size[1])+'.jpg', empty_face)
        draw_lms('empty_face_'+str(output_size[0])
                 + '_' + str(output_size[1])+'.jpg', tmp_pts)
    elif output_size[0] == 120 and output_size[1] == 120:
        tmp_pts = np.array([[42.29459953,
                             77.53179932,
                             60.02519989,
                             45.54930115,
                             74.72990036,
                             63.69630051,
                             63.50139999,
                             83.73660278,
                             104.3655014,
                             104.20410156]])
        empty_face = np.zeros((120, 120, 3))
        cv2.imwrite('empty_face_' + str(output_size[0])
                    + '_' + str(output_size[1]) + '.jpg', empty_face)
        draw_lms('empty_face_' + str(output_size[0])
                 + '_' + str(output_size[1]) + '.jpg', tmp_pts)

    default_square = True
    inner_padding_factor = 0.25
    outer_padding = (0, 0)
    # get the reference 5 landmarks position in the crop settings
    reference_5pts = get_reference_facial_points(
        output_size, inner_padding_factor, outer_padding, default_square)

    if (len(bbox) > 0):
        if type == 'labeled':
            # find the max bbox
            max_bb = []
            for box in bbox:
                x, y, r, b, _ = list(map(int, box))
                w = r - x + 1
                h = b - y + 1
                max_bb.append(w * h)
            index = max_bb.index(max(max_bb))
            facial5points = facial5points[[index]]
            facial5point = np.reshape(facial5points, (2, 5))
            dst_img = warp_and_crop_face(img, facial5point, reference_pts=reference_5pts, crop_size=output_size)
            dst_name = filename[: filename.rfind('.')]
            cv2.imwrite('{}_{}_mtcnn_aligned_{}x{}.jpg'.format(dst_name, type, output_size[0], output_size[1]), dst_img)
        elif type == 'unlabelled':
            for i in range(bbox.shape[0]):
                facial5point = np.reshape(facial5points[i], (2, 5))
                dst_img = warp_and_crop_face(img, facial5point, reference_pts=reference_5pts, crop_size=output_size)
                dst_name = filename[: filename.rfind('.')]
                cv2.imwrite('{}_{}_mtcnn_aligned_{}x{}_{}.jpg'.format(dst_name, type, output_size[0], output_size[1], i), dst_img)


if __name__ == "__main__":
    detector = MtcnnDetector()
    filename = './input.jpg'
    type = 'unlabelled' # labeled or unlabelled
    process(filename, type, output_size=(112, 112))