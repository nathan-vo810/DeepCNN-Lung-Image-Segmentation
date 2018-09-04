import os
import cv2
import pickle

DIR = os.path.dirname(__file__)
IMAGE_DIR = os.path.join(DIR, '../data/img/1.jpg')


def convert_grayscale(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def get_relative_location(x, y, image):
    image = cv2.imread(image)

    height, width = image.shape[:2]
    center_x, center_y = int(width / 2), int(height / 2)

    x_re = (x - center_x) / width
    y_re = (y - center_y) / height

    return x_re, y_re


def load_pkl(pkl_file):
    with open(pkl_file, 'rb') as file:
        locations = pickle.load(file)
    return locations


if __name__ == '__main__':
    locations = load_pkl('../data/99/validate/1/location.pkl')
    print('Done')
