from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

import os
import cv2
from src.utils import *
from scipy import misc

DIR = os.path.dirname(__file__)
WEIGHT_PATH = os.path.join(DIR, '../weight/weight.hdf5')
TEST_PATH = os.path.join(DIR, '../data/test/')


class VGGModel:
    def __init__(self, batch_size=128, epochs=20, lr=3e-4, load_weight=False):
        self.window_size = 224
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.load_weight = load_weight
        self.model = None

    def build_model(self):
        inputs = Input((self.window_size, self.window_size, 3))

        conv11 = Conv2D(filters=64, kernel_size=3, strides=1, padding="same", activation="relu")(inputs)
        conv12 = Conv2D(filters=64, kernel_size=3, strides=1, padding="same", activation="relu")(conv11)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv12)

        conv21 = Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation="relu")(pool1)
        conv22 = Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation="relu")(conv21)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv22)

        conv31 = Conv2D(filters=256, kernel_size=3, strides=1, padding="same", activation="relu")(pool2)
        conv32 = Conv2D(filters=256, kernel_size=3, strides=1, padding="same", activation="relu")(conv31)
        conv33 = Conv2D(filters=256, kernel_size=3, strides=1, padding="same", activation="relu")(conv32)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv33)

        conv41 = Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation="relu")(pool3)
        conv42 = Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation="relu")(conv41)
        conv43 = Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation="relu")(conv42)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv43)

        conv51 = Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation="relu")(pool4)
        conv52 = Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation="relu")(conv51)
        conv53 = Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation="relu")(conv52)
        pool5 = MaxPooling2D(pool_size=(2, 2))(conv53)

        flatten = Flatten()(pool5)
        fc1 = Dense(4096, activation='relu')(flatten)
        fc2 = Dense(1000, activation='relu')(fc1)
        output = Dense(1, activation='sigmoid')(fc2)

        vgg_model = Model(inputs=inputs, outputs=output)
        vgg_model.compile(optimizer=Adam(lr=self.lr), loss='binary_crossentropy', metrics=['accuracy'])

        self.model = vgg_model

    def train(self):
        print("Loading data...")
        train_data, label_data = load_data('generate', window_size=self.window_size)
        train_validate, label_validate = load_data('validate', window_size=self.window_size)
        print("Data loaded.")

        if self.load_weight:
            print("Loading weight...")
            self.model.load_weights(WEIGHT_PATH)
            print("Weight loaded.")

        print("Training...")
        model_checkpoint = ModelCheckpoint(WEIGHT_PATH, monitor='loss', verbose=1, save_best_only=True)
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
    network = VGGModel()
    network.build_model()
    train_history = network.train()
    plot_diagram(train_history)
