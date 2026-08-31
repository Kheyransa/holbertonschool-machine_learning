#!/usr/bin/env python3
"""Wasserstein GAN with weight clipping"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class WGAN_clip(keras.Model):
    """Wasserstein GAN with weight clipping"""

    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.005):
        """
        generator: generator network
        discriminator: discriminator network
        latent_generator: function generating latent vectors
        real_examples: tensor of real examples
        batch_size: size of the batches used in training
        disc_iter: number of discriminator training iterations
                   per generator training iteration
        learning_rate: learning rate for both optimizers
        """
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = .5
        self.beta_2 = .9

        # define the generator loss and optimizer:
        def gen_loss(x):
            return -tf.math.reduce_mean(x)

        self.generator.loss = gen_loss
        gen_optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2
        )
        self.generator.optimizer = gen_optimizer
        self.generator.compile(
            optimizer=self.generator.optimizer,
            loss=self.generator.loss
        )

        # define the discriminator loss and optimizer:
        def discr_loss(x, y):
            fake_mean = tf.math.reduce_mean(x)
            real_mean = tf.math.reduce_mean(y)
            return fake_mean - real_mean

        self.discriminator.loss = discr_loss
        discr_optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2
        )
        self.discriminator.optimizer = discr_optimizer
        self.discriminator.compile(
            optimizer=self.discriminator.optimizer,
            loss=self.discriminator.loss
        )

    # generator of fake samples of size batch_size
    def get_fake_sample(self, size=None, training=False):
        """Generates a fake sample using the generator"""
        if not size:
            size = self.batch_size
        latent_sample = self.latent_generator(size)
        return self.generator(latent_sample, training=training)

    # generator of real samples of size batch_size
    def get_real_sample(self, size=None):
        """Generates a real sample from real_examples"""
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    # overloading train_step()
    def train_step(self, useless_argument):
        """One training step: disc_iter discriminator updates
        (with weight clipping), followed by one generator update"""
        for _ in range(self.disc_iter):
            with tf.GradientTape() as disc_tape:
                real_sample = self.get_real_sample()
                fake_sample = self.get_fake_sample(training=True)

                disc_real = self.discriminator(real_sample, training=True)
                disc_fake = self.discriminator(fake_sample, training=True)

                discr_loss = self.discriminator.loss(disc_fake, disc_real)

            trainable_vars = self.discriminator.trainable_variables
            discr_gradients = disc_tape.gradient(
                discr_loss, trainable_vars)
            self.discriminator.optimizer.apply_gradients(
                zip(discr_gradients, trainable_vars))

            # clip the weights of the discriminator between -1 and 1
            for var in trainable_vars:
                clipped = tf.clip_by_value(var, -1, 1)
                var.assign(clipped)

        with tf.GradientTape() as gen_tape:
            fake_sample = self.get_fake_sample(training=True)
            disc_fake = self.discriminator(fake_sample, training=False)

            gen_loss = self.generator.loss(disc_fake)

        gen_trainable_vars = self.generator.trainable_variables
        gen_gradients = gen_tape.gradient(gen_loss, gen_trainable_vars)
        self.generator.optimizer.apply_gradients(
            zip(gen_gradients, gen_trainable_vars))

        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
