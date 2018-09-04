import os
import cv2
import time
from scipy import misc
import pickle as pkl

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPool2D, concatenate, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import to_categorical

from src.utils import load_data, hybrid_process, get_test_data

DIR = os.path.dirname(__file__)
WEIGHT_DIR = os.path.join(DIR, '../weight/')
TEST_DIR = os.path.join(DIR, '../data/test_data/')
RESULT_DIR = os.path.join(DIR, '../data/result/')
LOG_DIR = os.path.join(DIR, '../log/')


class HybridModel:
    def __init__(self):
        self.lr = 3e-4
        self.epochs = 100
        self.batch_size = 32
        self.model = self._build_model()

    def load_weight(self):
        print("Loading weight...")
        self.model.load_weights(WEIGHT_DIR + 'hybrid_0.904_loss_0.152.hdf5')
        print("Loaded.")

    def _build_model(self):
        window_input = Input(shape=(11, 11, 1))
        conv1 = Conv2D(filters=16, kernel_size=5, strides=1, padding='same', activation='relu')(window_input)

        conv2 = Conv2D(filters=24, kernel_size=3, strides=1, padding='same', activation='relu')(conv1)

        conv3 = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(conv2)
        pool3 = MaxPool2D(pool_size=(3, 3), strides=2)(conv3)

        conv4 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(pool3)
        pool4 = MaxPool2D(pool_size=(3, 3), strides=2)(conv4)

        conv5 = Conv2D(filters=96, kernel_size=3, strides=1, padding='same', activation='relu')(pool4)

        cnn_output = Flatten()(conv5)

        location_input = Input(shape=(2, 1))
        location_output = Dense(2, activation='relu')(location_input)
        location_output = Flatten()(location_output)

        concat = concatenate([cnn_output, location_output], axis=-1)
        fc = Dense(500, activation='relu')(concat)
        dropout = Dropout(0.5)(fc)
        output = Dense(2, activation='softmax')(dropout)

        model = Model(inputs=[window_input, location_input], outputs=output)
        model.compile(optimizer=Adam(lr=self.lr), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def fit(self, x_train, y_train, x_val, y_val):
        self.model.summary()
        y_train = to_categorical(y_train, num_classes=2)
        y_val = to_categorical(y_val, num_classes=2)

        weight_file = 'hybrid_{val_acc:.3f}_loss_{loss:.3f}.hdf5'
        model_checkpoint = ModelCheckpoint(WEIGHT_DIR + weight_file, monitor='val_acc', verbose=1, save_best_only=True)
        tensorboard = TensorBoard(log_dir=LOG_DIR + 'train_log', histogram_freq=0, write_graph=True, write_images=False)
        self.model.fit(x_train, y_train, validation_data=(x_val, y_val),
                       batch_size=self.batch_size, epochs=self.epochs, verbose=1, shuffle=True,
                       callbacks=[model_checkpoint, tensorboard])

    def predict(self, x_test, window_size):
        # print("Loading test image...")
        test_image = cv2.imread(os.path.join(TEST_DIR, x_test), 0)
        self.save(test_image, window_size)
        windows = self.load(window_size)
        # print("Loaded.")

        # print("Predicting...")
        predict_value = self.model.predict(windows, verbose=1)
        result_image = hybrid_process(predict_value, test_image, window_size)
        # print("Predicted.")

        # print("Saving...")
        result_path = os.path.join(RESULT_DIR, x_test)
        misc.imsave(result_path, result_image)
        # print("Saved.")

    def save(self, test_image, window_size):
        windows = get_test_data(test_image, window_size)
        with open('windows_{}.pkl'.format(str(window_size)), 'wb') as file:
            pkl.dump(windows, file)

    def load(self, window_size):
        with open('windows_{}.pkl'.format(str(window_size)), 'rb') as file:
            windows = pkl.load(file)
        return windows


def load_all_data(window_size):
    print("Loading training data...")
    x_train, y_train = load_data('generate', window_size)
    print("Loading validate data...")
    x_val, y_val = load_data('validate', window_size)
    print("Data loaded.")

    return x_train, y_train, x_val, y_val


if __name__ == '__main__':
    x_train, y_train, x_val, y_val = load_all_data(window_size=11)
    model = HybridModel()
    # model.load_weight()
    model.fit(x_train, y_train, x_val, y_val)
    # images = os.listdir(TEST_DIR)
    # images = [image for image in images if image.endswith('.jpg')]
    # start = time.process_time()
    # for i, image in enumerate(images):
    #     print("Processing {}/{}".format(i + 1, len(images)))
    #     model.predict(image, window_size=11)
    # print("Time eslapsed: {}".format(time.process_time() - start))
