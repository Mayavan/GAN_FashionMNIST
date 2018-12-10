from keras.layers import BatchNormalization, Activation
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers import Input, Dense, Reshape
from keras.models import Sequential, Model

import os


class Generator:
    def __init__(self):
        save_dir = os.path.join(os.path.dirname(os.getcwd()), 'saved_models')
        model_path = os.path.join(save_dir, "generator")

        self.channels = 1
        self.latent_dim = 100

        self.generator = self.createModel()

        self.generator.save(model_path)

    def createModel(self):
        model = Sequential()
        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((7, 7, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))
        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img, name='model_generator')


if __name__ == "__main__":
    Generator()
