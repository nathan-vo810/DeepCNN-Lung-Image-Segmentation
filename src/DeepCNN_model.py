from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint


class DeepCNN(object):
    """
    Deep CNN Implementation
    """

    def __init__(self, image_width=99, image_height=99):
        self.image_width = image_width
        self.image_height = image_height

    def build_model(self):
        inputs = Input((self.image_width, self.image_height))

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
        output = Dense(2, activation="softmax")(output)

        model = Model(inputs=inputs, outputs=output)

        return model
