import cv2
import os
import numpy as np
import random

DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(DIR, '../data/')

IMAGE_PATH = DATA_PATH + 'img/'
LABEL_PATH = DATA_PATH + 'label/'

SAVE_PATH = DATA_PATH + 'generate/'


class dataGenerator:
    def __init__(self, windowSize, numberOfWindows):
        self.windowSize = windowSize
        self.numberOfWindows = numberOfWindows

    def load_inputs(self, input_path, type='jpg'):
        inputs_list = []
        input_files = os.listdir(input_path)
        print("Loading: {}".format(input_path))
        input_files = [input_file for input_file in input_files if input_file.endswith(type)]

        for input_file in input_files:
            input_array = cv2.imread(input_path + input_file)
            inputs_list.append(input_array)
            cv2.waitKey(0)
        return inputs_list

    def padding_images(self, images):
        pad_images = []
        for i in range(len(images)):
            image_width = images[i].shape[0]
            image_height = images[i].shape[1]
            image_channels = images[i].shape[2]

            pad_width = self.windowSize + image_width + 1
            pad_height = self.windowSize + image_height + 1

            pad_image = np.zeros((pad_width, pad_height, image_channels))
            pad_image[:image_width, :image_height, :] = images[i]

            pad_images.append(pad_image)
            return pad_images

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
        print('Saving...')
        if is_lung:
            save_path = SAVE_PATH + '1/'
        else:
            save_path = SAVE_PATH + '0/'
        save_path = save_path + str(window_id) + '.jpg'
        cv2.imwrite(save_path, window)
        print('Saved {}.'.format(window_id))

    def generate_windows(self):
        print('Loading images...')
        images = self.load_inputs(input_path=IMAGE_PATH)
        labels = self.load_inputs(input_path=LABEL_PATH)
        print('Loaded.')

        width = images[0].shape[0]
        height = images[0].shape[1]

        print('Padding zeros...')
        images = self.padding_images(images)
        labels = self.padding_images(labels)
        print('Padded')

        image = images[0]
        label = labels[0]

        print(len(labels))

        # cv2.imshow('label', cv2.resize(image, (512, 512)))
        # cv2.waitKey(0)

        # print(label.shape)

        picked_points = np.ones((width, height))
        '''
        for i in range(self.numberOfWindows):
            x = random.randint(0, width)
            y = random.randint(0, height)
            if picked_points[x, y] == 1:
                is_lung = self.determine_label(x, y, label)
                window = self.get_window(x, y, image)
                self.save_window(window, is_lung, i)
        '''

if __name__ == "__main__":
    generator = dataGenerator(50, 10)
    generator.generate_windows()
