from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, SeparableConv2D, AveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from src import DataLoader
import matplotlib.pyplot as plt
import os

DIR = os.path.dirname(__file__)
WEIGHT_PATH = os.path.join(DIR, '../weight/weight.hdf5')


def load_data():
    data_loader = DataLoader()
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


class MobileNet:
    """
    Implementation of Mobile Net
    Network input size is 224x224x3
    """

    def __init__(self, batch_size=128, epochs=20, lr=3e-4, load_weight=False):
        self.image_size = (224, 224, 3)
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.load_weight = load_weight

    def build_model(self):
        inputs = Input(self.image_size)

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

        pool = AveragePooling2D(pool_size=(7, 7), strides=1)(dw9)
        flatten = Flatten()(pool)

        FC1 = Dense(1024, activation='relu')(flatten)
        output = Dense(1, activation='sigmoid')(FC1)

        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer=Adam(lr=self.lr), loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def train(self):
        model = self.build_model()
        model.summary()

        print("Loading data...")
        train_data, label_data = load_data()
        print("Data loaded.")

        if self.load_weight:
            print("Loading weight...")
            model.load_weights(WEIGHT_PATH)
            print("Weight loaded.")

        print("Training...")
        model_checkpoint = ModelCheckpoint(WEIGHT_PATH, monitor='loss', verbose=1, save_best_only=True)
        train_history = model.fit(train_data, label_data, batch_size=self.batch_size, epochs=20, verbose=1,
                                  validation_split=0.3,
                                  shuffle=True,
                                  callbacks=[model_checkpoint])
        return train_history


if __name__ == '__main__':
    network = MobileNet()
    train_history = network.train()
    plot_diagram(train_history)
