import cv2
import os
import numpy as np
import random

DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(DIR, '../data/')

IMAGE_PATH = DATA_PATH + 'img/'
LABEL_PATH = DATA_PATH + 'label/'

SAVE_PATH = DATA_PATH + 'generate/'


class DataGenerator:
    def __init__(self, windowSize, numberOfWindows):
        self.windowSize = windowSize
        self.numberOfWindows = numberOfWindows
        self.image_type = 'jpg'

    def get_files_list(self, input_path):
        files_list = os.listdir(input_path)
        files_list = [file for file in files_list if file.endswith(self.image_type)]
        return files_list

    def load_image(self, image_path):
        print("Loading: {}".format(image_path))
        image = cv2.imread(image_path)
        return image

    def pad_image(self, image):
        pad_width = image.shape[0] + self.windowSize + 1
        pad_height = image.shape[1] + self.windowSize + 1

        pad_image = np.zeros((pad_width, pad_height, image.shape[2]))
        pad_image[:image.shape[0], :image.shape[1], :] = image

        return pad_image

    def create_dir(self, image_name):
        save_dir = SAVE_PATH + image_name
        if not os.path.lexists(save_dir):
            os.makedirs(save_dir + '/0/')
            os.makedirs(save_dir + '/1/')

    def get_random_point(self, x_range, y_range):
        x = random.randint(0, x_range - 1)
        y = random.randint(0, y_range - 1)
        return x, y

    def determine_label(self, x, y, label):
        is_lung = True
        for i in range(self.windowSize):
            for j in range(self.windowSize):
                if label[x + i, y + j, 2] < 250:
                    is_lung = False
        return is_lung

    def get_window(self, x, y, image):
        window = np.zeros((self.windowSize, self.windowSize, 3))
        for i in range(self.windowSize):
            for j in range(self.windowSize):
                window[i, j, :] = image[x + i, y + j, :]
        return window

    def save_window(self, window, is_lung, window_id):
        if is_lung:
            save_path = SAVE_PATH + '/1/'
        else:
            save_path = SAVE_PATH + '/0/'
        save_path = save_path + window_id + '.jpg'
        cv2.imwrite(save_path, window)
        print('Saved {}.'.format(window_id))

    def generate_windows(self):
        # Get list of images and labels
        images_list = self.get_files_list(input_path=IMAGE_PATH)
        labels_list = self.get_files_list(input_path=LABEL_PATH)

        for i in range(len(images_list)):
            # Load and pad image
            image = self.load_image(IMAGE_PATH + images_list[i])
            width = image.shape[0]
            height = image.shape[1]
            image = self.pad_image(image)

            # Load label
            label = self.load_image(LABEL_PATH + labels_list[i])
            label = self.pad_image(label)

            # Pick random points, check label and save
            picked_points = np.ones((width, height))
            count = 0
            while count < self.numberOfWindows:
                x, y = self.get_random_point(width, height)
                if picked_points[x, y] == 1:
                    count += 1
                    picked_points[x, y] = 0
                    is_lung = self.determine_label(x, y, label)
                    lung_window = self.get_window(x, y, image)
                    window_id = images_list[i].split('.')[0] + '_' + str(count)
                    self.save_window(lung_window, is_lung, window_id)


if __name__ == "__main__":
    generator = DataGenerator(99, 1000)
    generator.generate_windows()
