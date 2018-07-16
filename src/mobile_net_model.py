from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, SeparableConv2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

import os
import cv2
from scipy import misc
from src.utils import *

DIR = os.path.dirname(__file__)
WEIGHT_PATH = os.path.join(DIR, '../weight/weight_mobile_net.hdf5')
TEST_PATH = os.path.join(DIR, '../data/test/')


class MobileNet:
    """
    Implementation of Mobile Net
    Network input size is 224x224x3
    """

    def __init__(self, batch_size=128, epochs=6, lr=1e-3, load_weight=False):
        self.window_size = 224
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.load_weight = load_weight

    def build_model(self):
        inputs = Input((self.window_size, self.window_size, 3))

        conv = Conv2D(filters=32, kernel_size=3, strides=2, padding='same', activation='relu')(inputs)
        dw1 = SeparableConv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(conv)
        dw2 = SeparableConv2D(filters=128, kernel_size=3, strides=2, padding='same', activation='relu')(dw1)
        dw3 = SeparableConv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu')(dw2)
        dw4 = SeparableConv2D(filters=256, kernel_size=3, strides=2, padding='same', activation='relu')(dw3)
        dw5 = SeparableConv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu')(dw4)
        dw6 = SeparableConv2D(filters=512, kernel_size=3, strides=2, padding='same', activation='relu')(dw5)

        dw7 = SeparableConv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu')(dw6)
        dw7 = SeparableConv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu')(dw7)
        dw7 = SeparableConv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu')(dw7)
        dw7 = SeparableConv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu')(dw7)
        dw7 = SeparableConv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu')(dw7)

        dw8 = SeparableConv2D(filters=1024, kernel_size=3, strides=2, padding='same', activation='relu')(dw7)
        dw9 = SeparableConv2D(filters=1024, kernel_size=3, strides=1, padding='same', activation='relu')(dw8)

        pool = GlobalAveragePooling2D()(dw9)
        # flatten = Flatten()(pool)

        # fc1 = Dense(1024, activation='relu')(flatten)
        output = Dense(1, activation='sigmoid')(pool)

        mobile_net_model = Model(inputs=inputs, outputs=output)
        mobile_net_model.compile(optimizer=Adam(lr=self.lr), loss='binary_crossentropy', metrics=['accuracy'])

        return mobile_net_model

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
        model = self.build_model()
        model.summary()
        print("Loading weight...")
        model.load_weights(WEIGHT_PATH)
        print("Loaded.")

        print("Loading test image...")
        test_image = cv2.imread(image_path)
        windows = generate_test(test_image, self.window_size)
        print("Loaded.")

        print("Predicting...")
        predict_value = model.predict(windows, verbose=1)
        print(predict_value)
        result_image = process_predict_value(predict_value, 0.8, test_image, self.window_size)
        print("Predicted.")

        print("Saving...")
        misc.imsave(TEST_PATH + 'result.jpg', result_image)
        print("Saved.")


if __name__ == '__main__':
    network = MobileNet()
    # network.build_model()
    # train_history = network.train()
    # plot_diagram(train_history)
    network.predict(TEST_PATH + '21.jpg')
