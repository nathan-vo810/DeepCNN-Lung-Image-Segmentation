import matplotlib.pyplot as plt
import numpy as np
from src.data_loader import DataLoader
from src.data_generator import DataGenerator


def load_data(data_path, window_size):
    data_loader = DataLoader(data_path, window_size)
    train_data, label_data = data_loader.load_train_data()
    return train_data, label_data


def plot_diagram(history):
    # Loss Curves
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['loss'], 'r', linewidth=3.0)
    plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Loss Curves', fontsize=16)

    # Accuracy Curves
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['acc'], 'r', linewidth=3.0)
    plt.plot(history.history['val_acc'], 'b', linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Accuracy Curves', fontsize=16)


def generate_test(test_image, window_size):
    data_gen = DataGenerator(window_size)
    windows = data_gen.get_test_data(test_image)
    return windows


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
