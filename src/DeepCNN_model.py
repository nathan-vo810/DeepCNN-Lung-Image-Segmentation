import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from src.utils import *
import os
import cv2
from scipy import misc
import pickle as pkl

DIR = os.path.dirname(__file__)
WEIGHT_DIR = os.path.join(DIR, '../weight/')
weight_11_file = '11_weight_deep_cnn_val_0.762_loss_0.464.hdf5'
weight_99_file = '99_weight_deep_cnn_val_acc_92.hdf5'
TEST_DIR = os.path.join(DIR, '../data/test/')
WINDOW_SIZE = 11


class DeepCNN:
    """
    Deep CNN Implementation
    """

    def __init__(self, load_weight=False, batch_size=32, epochs=100, lr=1e-4):
        # self.input_shape = (WINDOW_SIZE, WINDOW_SIZE, 3)
        self.load_weight = load_weight
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr

    def build_model_window_11(self, weight_file=None):
        '''
        Window size = 11 -> Model: Conv1 -> Conv2 -> Conv3 -> Pool3 -> Conv4 -> Pool4 -> Conv5 -> FC1 -> Sigmoid
        '''
        model = Sequential()
        model.add(Conv2D(filters=16, kernel_size=5, strides=1, padding='valid', activation='relu',
                         input_shape=(11, 11, 3), name='Conv1'))

        model.add(Conv2D(filters=24, kernel_size=3, strides=1, padding='same', activation='relu', name='Conv2'))

        model.add(Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu', name='Conv3'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', name='Conv4'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(filters=96, kernel_size=3, strides=1, padding='same', activation='relu', name='Conv5'))

        model.add(Flatten())
        model.add(Dense(500, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer=Adam(lr=self.lr), loss='binary_crossentropy', metrics=['accuracy'])

        if weight_file:
            print("Loading weight...")
            model.load_weights(WEIGHT_DIR + weight_file)
            print("Weight loaded.")

        return model

    def build_model_window_99(self, weight_file=None):
        '''
        Window size = 32,99 -> Model: (Conv -> Pool) x 4 -> Conv5 -> FC1 -> Sigmoid
        '''
        model = Sequential()
        model.add(Conv2D(filters=16, kernel_size=5, strides=1, padding='valid', activation='relu',
                         input_shape=(99, 99, 3), name='Conv1'))
        # model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(filters=24, kernel_size=3, strides=1, padding='same', activation='relu', name='Conv2'))
        # model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu', name='Conv3'))
        # model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', name='Conv4'))
        # model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(filters=96, kernel_size=3, strides=1, padding='same', activation='relu', name='Conv5'))
        # model.add(BatchNormalization())

        model.add(Flatten())
        # Only use Dense 1000 for window size 32 and val_acc 0.871
        # model.add(Dense(1000, activation='relu'))
        # model.add(Dropout(0.2))
        model.add(Dense(500, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer=Adam(lr=self.lr), loss='binary_crossentropy', metrics=['accuracy'])

        if weight_file:
            print("Loading weight...")
            model.load_weights(WEIGHT_DIR + weight_file)
            print("Weight loaded.")

        return model

    def fit(self, x_train, y_train, x_val, y_val):
        if WINDOW_SIZE == 11:
            model = self.build_model_window_11()
        else:
            model = self.build_model_window_99()
        print("Training...")
        weight_file = str(WINDOW_SIZE) + '_weight_deep_cnn_val_{val_acc:.3f}_loss_{loss:.3f}.hdf5'
        model_checkpoint = ModelCheckpoint(WEIGHT_DIR + weight_file, monitor='val_acc', verbose=1, save_best_only=True)
        history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                            batch_size=self.batch_size, epochs=self.epochs, verbose=1, shuffle=True,
                            callbacks=[model_checkpoint])

        return history

    def prepare_models(self):
        model_11 = self.build_model_window_11()
        model_99 = self.build_model_window_99()

        print("Loading weight...")
        model_11.load_weights(WEIGHT_DIR + weight_11_file)
        model_99.load_weights(WEIGHT_DIR + weight_99_file)
        print("Loaded.")
        return model_11, model_99

    def predict(self, image_path):
        model = self.build_model_window_11()
        model.summary()
        print("Loading weight...")
        model.load_weights(WEIGHT_DIR + weight_11_file)
        print("Loaded.")

        print("Loading test image...")
        test_image = cv2.imread(image_path)
        windows = generate_test(test_image, WINDOW_SIZE)
        print("Loaded.")

        print("Predicting...")
        predict_value = model.predict(windows, verbose=1)
        result_image = process_predict_value(predict_value, 0.5, test_image, WINDOW_SIZE)
        print("Predicted.")

        print("Saving...")
        misc.imsave(TEST_DIR + 'result.jpg', result_image)
        print("Saved.")

    def predict_custom(self, image_path):
        model_11, model_99 = self.prepare_models()

        print("Loading test image...")
        test_image = cv2.imread(image_path)

        '''
        windows_99 = generate_test(test_image, 99)
        '''

        # with open('list.pkl', 'wb') as file:
        #     pkl.dump(windows_99,file, pkl.HIGHEST_PROTOCOL)

        with open('list.pkl', 'rb') as file:
            windows_99 = pkl.load(file)
        print("Loaded.")

        print("Predicting...")

        predict_value_99 = model_99.predict(windows_99, verbose=1)

        width, height = test_image.shape[:2]
        n_cols, n_rows = int(width / 99), int(height / 99)
        result_image = np.array([]).reshape(0, n_cols * 99)

        row = np.array([]).reshape((99, 0))
        for i in range(len(windows_99)):
            windows_11 = generate_test(windows_99[i], window_size=11)
            predict_value_11 = model_11.predict(windows_11, verbose=1)
            predict_value_11 *= predict_value_99[i]
            predicted_window = process_predict_value(predict_value_11, 0.8, windows_99[i], 11)
            row = np.hstack((row, predicted_window))
            if (i + 1) % n_cols == 0:
                result_image = np.vstack((result_image, row))
                row = np.array([]).reshape((99, 0))

        print("Predicted.")

        print("Saving...")
        misc.imsave(TEST_DIR + 'result.jpg', result_image)
        print("Saved.")


def load_all_data(window_size):
    print("Loading training data...")
    x_train, y_train = load_data('generate', window_size)
    print("Loading validate data...")
    x_val, y_val = load_data('validate', window_size)
    print("Data loaded.")

    return x_train, y_train, x_val, y_val


if __name__ == '__main__':
    network = DeepCNN()

    # x_train, y_train, x_val, y_val = load_all_data(window_size=WINDOW_SIZE)
    # train_history = network.fit(x_train, y_train, x_val, y_val)
    # plot_diagram(train_history)

    network.predict(TEST_DIR + '21.jpg')
