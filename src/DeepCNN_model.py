import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from src.utils import *
import os
import cv2
from scipy import misc

DIR = os.path.dirname(__file__)
WEIGHT_PATH = os.path.join(DIR, '../weight/')
TEST_PATH = os.path.join(DIR, '../data/test/')


class DeepCNN:
    """
    Deep CNN Implementation
    """

    def __init__(self, window_size=11, load_weight=False, batch_size=32, epochs=100, lr=1e-4):
        self.window_size = window_size
        self.load_weight = load_weight
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr

    def build_model(self):
        inputs = Input((self.window_size, self.window_size, 3))

        conv1 = Conv2D(filters=16, kernel_size=5, strides=1, padding="valid", activation="relu")(inputs)
        # bn1 = BatchNormalization()(conv1)
        # pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(filters=24, kernel_size=3, strides=1, padding="same", activation="relu")(conv1)
        # bn2 = BatchNormalization()(conv2)
        # pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(filters=32, kernel_size=3, strides=1, padding="same", activation="relu")(conv2)
        # bn3 = BatchNormalization()(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(filters=64, kernel_size=3, strides=1, padding="same", activation="relu")(pool3)
        # bn4 = BatchNormalization()(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(filters=96, kernel_size=3, strides=1, padding="same", activation="relu")(pool4)
        # bn5 = BatchNormalization()(conv5)

        flatten = Flatten()(conv5)
        fc1 = Dense(500, activation="relu")(flatten)
        fc1 = Dropout(0.2)(fc1)
        fc2 = Dense(500, activation="relu")(fc1)
        fc2 = Dropout(0.2)(fc2)
        output = Dense(1, activation="sigmoid")(fc2)

        model = Model(inputs=inputs, outputs=output)

        model.compile(optimizer=Adam(lr=self.lr), loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def train(self):
        model = self.build_model()
        print("Loading training data...")
        train_data, label_data = load_data('generate', window_size=self.window_size)
        print("Loading validate data...")
        train_validate, label_validate = load_data('validate', window_size=self.window_size)
        print("Data loaded.")

        if self.load_weight:
            model.load_weights(WEIGHT_PATH + '32_weight_deep_cnn_val_0.859_loss_0.274.hdf5')
            print("Weight loaded.")

        print("Training...")
        weight_file = '32_weight_deep_cnn_val_{val_acc:.3f}_loss_{loss:.3f}.hdf5'
        model_checkpoint = ModelCheckpoint(WEIGHT_PATH + weight_file, monitor='val_acc', verbose=1, save_best_only=True)
        history = model.fit(train_data, label_data, validation_data=(train_validate, label_validate),
                                 batch_size=self.batch_size, epochs=self.epochs, verbose=1, shuffle=True,
                                 callbacks=[model_checkpoint])

        return history

    def predict(self, image_path):
        model = self.build_model()
        print("Loading weight...")
        model.load_weights(WEIGHT_PATH + '11_weight_deep_cnn_val_0.762_loss_0.464.hdf5')
        print("Loaded.")

        print("Loading test image...")
        test_image = cv2.imread(image_path)
        windows = generate_test(test_image, self.window_size)
        print("Loaded.")

        print("Predicting...")
        predict_value = model.predict(windows, verbose=1)
        result_image = process_predict_value(predict_value, 0.7, test_image, self.window_size)
        print("Predicted.")

        print("Saving...")
        misc.imsave(TEST_PATH + 'result.jpg', result_image)
        print("Saved.")


if __name__ == '__main__':
    network = DeepCNN()
    train_history = network.train()
    plot_diagram(train_history)
    # network.predict(TEST_PATH + '7.jpg')
