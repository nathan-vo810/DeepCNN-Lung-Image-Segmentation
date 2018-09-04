import cv2
import os
import numpy as np

from src.utils import get_relative_location, get_random_point, save_window, determine_label_center, pad_image, \
    get_list_files, get_window, save_locations
from src.preprocess import convert_grayscale

DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(DIR, '../data/')

IMAGE_PATH = DATA_PATH + 'img/'
LABEL_PATH = DATA_PATH + 'label/'

VAL_IMAGE_PATH = DATA_PATH + 'img_val/'
VAL_LABEL_PATH = DATA_PATH + 'label_val/'

TRAIN_DATA_PATH = DATA_PATH + '11/generate/'
VAL_DATA_PATH = DATA_PATH + '11/validate/'


class DataGenerator:
    def __init__(self, window_size):
        self.window_size = window_size
        self.image_type = 'jpg'

    def generate_windows(self, is_lung, image_path, label_path, save_dir, number_of_windows):
        # Get list of images and labels
        images_list = get_list_files(input_path=image_path, file_type=self.image_type)
        labels_list = get_list_files(input_path=label_path, file_type=self.image_type)

        locations = []

        for i in range(len(images_list)):
            # Load and pad image
            image = cv2.imread(IMAGE_PATH + images_list[i])
            image = convert_grayscale(image)
            height, width = image.shape[:2]
            image = pad_image(image, padding=self.window_size)

            # Load label
            label = cv2.imread(LABEL_PATH + labels_list[i])
            label = convert_grayscale(label)
            label = pad_image(label, padding=self.window_size)

            # Pick random points, check label and save
            picked_points = np.ones((height, width))
            count = 0
            while count < number_of_windows:
                x, y = get_random_point(height, width)
                if picked_points[x, y] == 1:
                    picked_points[x, y] = 0
                    if is_lung == determine_label_center(x, y, label, self.window_size):
                        count += 1
                        x_re, y_re = get_relative_location(x, y, height, width)
                        locations.append((x_re, y_re))
                        window = get_window(x, y, image, self.window_size)
                        window_id = images_list[i].split('.')[0] + '_' + str(count)
                        save_window(window, is_lung, window_id, save_dir)

        save_locations(locations, is_lung, save_dir)


if __name__ == "__main__":
    generator = DataGenerator(window_size=11)
    generator.generate_windows(True, IMAGE_PATH, LABEL_PATH, TRAIN_DATA_PATH, 3000)
    generator.generate_windows(False, IMAGE_PATH, LABEL_PATH, TRAIN_DATA_PATH, 3000)
    generator.generate_windows(True, VAL_IMAGE_PATH, VAL_LABEL_PATH, VAL_DATA_PATH, 1000)
    generator.generate_windows(False, VAL_IMAGE_PATH, VAL_LABEL_PATH, VAL_DATA_PATH, 1000)
