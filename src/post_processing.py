import cv2
import os
import argparse
import numpy as np

DIR = os.path.dirname(__file__)
IMAGE_DIR = os.path.join(DIR, '../data/test_data')
LABEL_DIR = os.path.join(DIR, '../data/result')
OUTPUT_DIR = os.path.join(DIR, '../data/processed')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', dest='image', default='7.jpg')
    parser.add_argument('--label', dest='label', default='result_11.jpg')

    return parser.parse_args()


def cut_lung(image, label, name):
    height, width = image.shape[:2]

    scale_factor = 5

    image = cv2.resize(image, (0, 0), fx=1 / scale_factor, fy=1 / scale_factor)
    label = cv2.resize(label, (0, 0), fx=1 / scale_factor, fy=1 / scale_factor)

    scaled_h = int(height / scale_factor)
    scaled_w = int(width / scale_factor)

    result = np.zeros((scaled_h, scaled_w, image.shape[2]))
    for i in range(scaled_h):
        for j in range(scaled_w):
            print('Processing {} - {}'.format(i, j))
            if label[i, j, 2] > 10:
                result[i, j, :] = image[i, j, :]
            else:
                result[i, j, :] = 255

    result = cv2.resize(result, (width, height))

    result_path = os.path.join(OUTPUT_DIR, name)
    cv2.imwrite(result_path, result)


def open_close_morphology(image):
    kernel = np.ones((13, 13), np.uint8)

    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    return image


def main(args):
    images = os.listdir(IMAGE_DIR)
    images = sorted([image for image in images if image.endswith('jpg')])

    for f in images:
        print("Processing: {}".format(f))
        image = cv2.imread(os.path.join(IMAGE_DIR, f))
        label = cv2.imread(os.path.join(LABEL_DIR, f))

        label = open_close_morphology(label)

        cut_lung(image, label, f)


if __name__ == '__main__':
    main(parse_args())
