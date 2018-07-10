from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from src.utils import *
import os
import cv2
from scipy import misc

DIR = os.path.dirname(__file__)
WEIGHT_PATH = os.path.join(DIR, '../weight/weight_deep_cnn.hdf5')
TEST_PATH = os.path.join(DIR, '../data/test/')


class DeepCNN:
    """
    Deep CNN Implementation
    """

    def __init__(self, window_size=32, load_weight=False, batch_size=128, epochs=20, lr=3e-4):
        self.window_size = window_size
        self.load_weight = load_weight
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.model = None

    def build_model(self):
        inputs = Input((self.window_size, self.window_size, 3))

        conv1 = Conv2D(filters=16, kernel_size=3, strides=1, padding="same", activation="relu")(inputs)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(filters=24, kernel_size=3, strides=1, padding="same", activation="relu")(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(filters=32, kernel_size=3, strides=1, padding="same", activation="relu")(pool2)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(filters=64, kernel_size=3, strides=1, padding="same", activation="relu")(pool3)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(filters=96, kernel_size=3, strides=1, padding="same", activation="relu")(pool4)

        flatten = Flatten()(conv5)
        fc1 = Dense(500, activation="relu")(flatten)
        fc1 = Dropout(0.5)(fc1)
        output = Dense(1, activation="sigmoid")(fc1)

        model = Model(inputs=inputs, outputs=output)

        model.compile(optimizer=Adam(lr=self.lr), loss='binary_crossentropy', metrics=['accuracy'])

        self.model = model

    def train(self):
        print("Loading data...")
        train_data, label_data = load_data('generate', window_size=self.window_size)
        train_validate, label_validate = load_data('validate', window_size=self.window_size)
        print("Data loaded.")

        if self.load_weight:
            self.model.load_weights(WEIGHT_PATH)
            print("Weight loaded.")

        print("Training...")
        model_checkpoint = ModelCheckpoint(WEIGHT_PATH, monitor='val_acc', verbose=1, save_best_only=True)
        history = self.model.fit(train_data, label_data, validation_data=(train_validate, label_validate),
                                 batch_size=self.batch_size, epochs=self.epochs, verbose=1, shuffle=True,
                                 callbacks=[model_checkpoint])

        return history

    def predict(self, image_path):
        print("Loading weight...")
        self.model.load_weights(WEIGHT_PATH)
        print("Loaded.")

        print("Loading test image...")
        test_image = cv2.imread(image_path)
        windows = generate_test(test_image, self.window_size)
        print("Loaded.")

        print("Predicting...")
        predict_value = self.model.predict(windows, verbose=1)
        result_image = process_predict_value(predict_value, 0.8, test_image, self.window_size)
        print("Predicted.")

        print("Saving...")
        misc.imsave(TEST_PATH + 'result.jpg', result_image)
        print("Saved.")


if __name__ == '__main__':
    network = DeepCNN()
    network.build_model()
    # train_history = network.train()
    # plot_diagram(train_history)
    network.predict(TEST_PATH + '21.jpg')
