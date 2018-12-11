from __future__ import print_function, division
from keras.models import load_model
from keras.datasets import fashion_mnist
from keras.layers import Input

from keras.models import Model
from keras.optimizers import Adam
import os
import matplotlib.pyplot as plt

import numpy as np
from matplotlib import pyplot

class DCGAN:
    def __init__(self, D, G):
        self.model_dir = os.path.join(os.path.dirname(os.getcwd()), 'saved_models')
        model_path_discriminator = os.path.join(self.model_dir, D)
        model_path_generator = os.path.join(self.model_dir, G)

        self.image_dir = os.path.join(os.path.dirname(os.getcwd()), 'generated_images')

        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = load_model(model_path_discriminator)
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = load_model(model_path_generator)

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated_images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        (X_train, _), (_, _) = fashion_mnist.load_data()

        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        D_loss = []
        G_loss = []
        D_accuracy = []
        for epoch in range(epochs+1):
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            D_loss.append(d_loss[0])
            D_accuracy.append(d_loss[1])
            G_loss.append(g_loss)

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

        self.plot_metrics(D_loss, D_accuracy, G_loss, batch_size, epochs, 50)
        self.discriminator.save(os.path.join(self.model_dir, "discriminator_%d" % epochs))
        self.generator.save(os.path.join(self.model_dir, "generator_%d" % epochs))

    def plot_metrics(self, D_loss, D_accuracy, G_loss, batch_size, epochs, graph_batch):
        n = graph_batch
        D_loss2 = [sum(D_loss[i:i + n]) / n for i in range(0, len(D_loss), n)]
        D_accuracy2 = [sum(D_accuracy[i:i + n]) / n for i in range(0, len(D_accuracy), n)]
        G_loss2 = [sum(G_loss[i:i + n]) / n for i in range(0, len(G_loss), n)]

        for i in range(0, len(D_loss)):
            D_loss[i] = D_loss2[i // n]
            D_accuracy[i] = D_accuracy2[i // n]
            G_loss[i] = G_loss2[i // n]

        epoch = np.arange(0, epochs, 1)
        # plot metrics
        pyplot.title(''.join(["Discriminator vs Generator loss with batch size of ", str(batch_size)]))
        pyplot.xlabel('Epochs')
        pyplot.ylabel('Loss')

        pyplot.plot(epoch, D_loss[:-1], label='Discriminator Loss')
        pyplot.plot(epoch, G_loss[:-1], label='Generator Loss')
        pyplot.legend()
        pyplot.savefig(''.join([os.path.join(os.path.dirname(os.getcwd()), 'results'), '/loss.png']))
        pyplot.show()

        pyplot.title(''.join(["Discriminator Accuracy with batch size of ", str(batch_size)]))
        pyplot.xlabel('Epochs')
        pyplot.ylabel('Accuracy')
        pyplot.plot(D_accuracy[:-1], label='Discriminator Accuracy')
        pyplot.legend()
        pyplot.savefig(''.join([os.path.join(os.path.dirname(os.getcwd()), 'results'), '/Accuracy.png']))
        pyplot.show()

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(os.path.join(self.image_dir, "mnist_%d.png" % epoch))
        plt.close()


if __name__ == '__main__':
    dcgan = DCGAN("discriminator", "generator")
    dcgan.train(epochs=10000, batch_size=32, save_interval=2000)
