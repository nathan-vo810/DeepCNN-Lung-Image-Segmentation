import os
import cv2
import pickle

DIR = os.path.dirname(__file__)
IMAGE_DIR = os.path.join(DIR, '../data/img/1.jpg')


def convert_grayscale(image):
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def get_relative_location(x, y, image):
    image = cv2.imread(image)
    image = cv2.rectangle(image, (x, y), (x + 11, y + 11), (0, 255, 0), 1)

    height, width = image.shape[:2]
    center_x, center_y = int(width / 2), int(height / 2)
    image = cv2.line(image, (center_x, 0), (center_x, height), (255, 0, 0), 3)
    image = cv2.line(image, (0, center_y), (width, center_y), (255, 0, 0), 3)

    x_re = (x - center_x) / width
    y_re = (y - center_y) / height

    print(width, height, x_re, y_re)
    return x_re, y_re


def load_pkl(pkl_file):
    with open(pkl_file, 'rb') as file:
        locations = pickle.load(file)
    return locations


if __name__ == '__main__':
    locations = load_pkl('../data/99/validate/1/location.pkl')
    print('Done')
