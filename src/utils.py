import os
import cv2
import random
import pickle
import numpy as np
from src.data_loader import DataLoader


def get_window(x, y, image, window_size):
    window = np.zeros((window_size, window_size))
    for i in range(window_size):
        for j in range(window_size):
            window[i, j] = image[x + i, y + j]
    return window


def save_window(window, is_lung, window_id, save_dir):
    path = os.path.join(save_dir, str(int(is_lung)), window_id + '.jpg')
    cv2.imwrite(path, window)
    print('Saved {}.'.format(window_id))


def get_relative_location(x, y, height, width):
    x_re = (x - (width / 2)) / width
    y_re = (y - (height / 2)) / height
    return x_re, y_re


def save_locations(locations, is_lung, save_dir):
    locations = np.array(locations)
    locations = locations.ravel()
    locations = locations.reshape((-1, 2))
    location_path = os.path.join(save_dir, str(int(is_lung)) + '/location.pkl')
    with open(location_path, 'wb') as f:
        pickle.dump(locations, f)


def get_random_point(x_range, y_range):
    x = random.randint(0, x_range - 1)
    y = random.randint(0, y_range - 1)
    return x, y


def determine_label(x, y, label, window_size):
    is_lung = True
    for i in range(window_size):
        for j in range(window_size):
            if label[x + i, y + j, 2] < 250:
                is_lung = False
    return is_lung


def determine_label_center(x, y, label, window_size):
    is_lung = True
    center_x = x + int(window_size / 2)
    center_y = y + int(window_size / 2)
    if label[center_x, center_y] < 250:
        is_lung = False
    return is_lung


def pad_image(image, padding):
    print(image.shape)
    pad_height = image.shape[0] + padding + 1
    pad_width = image.shape[1] + padding + 1

    pad_image = np.zeros((pad_height, pad_width))
    pad_image[:image.shape[0], :image.shape[1]] = image

    return pad_image


def get_list_files(input_path, file_type):
    files_list = os.listdir(input_path)
    files_list = [file for file in files_list if file.endswith(file_type)]
    return sorted(files_list)


def load_data(data_path, window_size):
    data_loader = DataLoader(data_path, window_size)
    train_data, label_data = data_loader.load_train_data()
    return train_data, label_data


def get_test_data(image, window_size):
    img_height, img_width = image.shape[:2]
    image = pad_image(image, padding=window_size)

    n_cols = int(img_width / window_size)
    n_rows = int(img_height / window_size)
    windows = np.empty(
        (n_rows * n_cols, window_size, window_size, 1))

    index = 0
    locations = []
    for i in range(n_rows):
        for j in range(n_cols):
            # print("Window number {}".format(index + 1))
            window = get_window(i * window_size, j * window_size, image, window_size)
            window = np.reshape(window, (window.shape[0], window.shape[1], -1))
            windows[index] = window
            x_re, y_re = get_relative_location(i * window_size, j * window_size, img_height, img_width)
            locations.append((x_re, y_re))
            index += 1
    locations = np.array(locations)
    locations.ravel()
    locations = locations.reshape((locations.shape[0], -1, 1))
    return [windows, locations]


def hybrid_process(predict_value, test_image, window_size):
    predict_value = predict_value.argmax(axis=1)
    predict_value[predict_value == 1] = 255

    result_image = get_result_label(test_image, predict_value, window_size)
    return result_image


def process_predict_value(predict_value, threshold, test_image, window_size):
    predict_value[predict_value < threshold] = 0
    predict_value[predict_value >= threshold] = 255

    result_image = get_result_label(test_image, predict_value, window_size)
    return result_image


def get_result_label(test_image, predict_value, window_size):
    image_height, image_width = test_image.shape[:2]
    result_image = np.empty((image_height, image_width))

    n_rows = int(image_height / window_size)
    n_cols = int(image_width / window_size)

    predict_value = predict_value.reshape((n_rows, n_cols))

    for i in range(n_rows):
        for j in range(n_cols):
            fill_window(predict_value[i, j], i, j, result_image, window_size)
    return result_image


def fill_window(value, i, j, window, window_size):
    for x in range(window_size):
        for y in range(window_size):
            window[i * window_size + x, j * window_size + y] = value
