from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from src import DataLoader
import matplotlib.pyplot as plt
import os

DIR = os.path.dirname(__file__)
WEIGHT_PATH = os.path.join(DIR, '../weight/weight.hdf5')


class DeepCNN:
    """
    Deep CNN Implementation
    """

    def __init__(self, image_width=99, image_height=99):
        self.image_width = image_width
        self.image_height = image_height

    def load_data(self):
        data_loader = DataLoader()
        train_data, label_data = data_loader.load_train_data()
        return train_data, label_data

    def build_model(self):
        inputs = Input((self.image_width, self.image_height, 3))

        conv1 = Conv2D(filters=16, kernel_size=5, strides=2, padding="valid", activation="relu")(inputs)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(filters=24, kernel_size=3, strides=1, padding="same", activation="relu")(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(filters=32, kernel_size=3, strides=1, padding="same", activation="relu")(pool2)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(filters=64, kernel_size=3, strides=1, padding="same", activation="relu")(pool3)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(filters=96, kernel_size=3, strides=1, padding="same", activation="relu")(pool4)

        output = Flatten()(conv5)
        output = Dense(500, activation="relu")(output)
        output = Dense(1, activation="sigmoid")(output)

        model = Model(inputs=inputs, outputs=output)

        model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def train(self):
        model = self.build_model()
        model.summary()
        print("Loading data...")
        train_data, label_data = self.load_data()
        print("Data loaded.")
        # print("Loading weight...")
        # model.load_weights(WEIGHT_PATH)
        # print("Weight loaded.")
        print("Training...")
        model_checkpoint = ModelCheckpoint(WEIGHT_PATH, monitor='loss', verbose=1, save_best_only=True)
        train_history = model.fit(train_data, label_data, batch_size=128, epochs=20, verbose=1, validation_split=0.3, shuffle=True,
                  callbacks=[model_checkpoint])
        return train_history

    def plot_diagram(self, history):
        # Loss Curves
        plt.figure(figsize=[8,6])
        plt.plot(history.history['loss'],'r',linewidth=3.0)
        plt.plot(history.history['val_loss'],'b',linewidth=3.0)
        plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
        plt.xlabel('Epochs ',fontsize=16)
        plt.ylabel('Loss',fontsize=16)
        plt.title('Loss Curves',fontsize=16)
        
        # Accuracy Curves
        plt.figure(figsize=[8,6])
        plt.plot(history.history['acc'],'r',linewidth=3.0)
        plt.plot(history.history['val_acc'],'b',linewidth=3.0)
        plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
        plt.xlabel('Epochs ',fontsize=16)
        plt.ylabel('Accuracy',fontsize=16)
        plt.title('Accuracy Curves',fontsize=16)


if __name__ == '__main__':
    network = DeepCNN()
    train_history = network.train()
    network.plot_diagram(train_history)