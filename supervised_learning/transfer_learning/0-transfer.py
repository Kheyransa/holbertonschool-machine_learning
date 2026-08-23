#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras as K


def preprocess_data(X, Y):
    """
    Pre-processes the data for the model.

    Args:
        X: numpy.ndarray of shape (m, 32, 32, 3)
        Y: numpy.ndarray of shape (m,)

    Returns:
        X_p: preprocessed images
        Y_p: one-hot encoded labels
    """
    X_p = K.applications.resnet50.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)

    return X_p, Y_p


if __name__ == "__main__":
    # Load CIFAR-10
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()

    # Preprocess data
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    # Load pretrained ResNet50 without its classification head
    base_model = K.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3)
    )

    # Freeze pretrained layers
    base_model.trainable = False

    # Build model
    inputs = K.Input(shape=(32, 32, 3))

    x = K.layers.Lambda(
        lambda image: tf.image.resize(image, (224, 224))
    )(inputs)

    x = base_model(x, training=False)

    x = K.layers.GlobalAveragePooling2D()(x)

    x = K.layers.Dense(256, activation="relu")(x)
    x = K.layers.Dropout(0.5)(x)

    outputs = K.layers.Dense(10, activation="softmax")(x)

    model = K.Model(inputs, outputs)

    # Compile
    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Train
    model.fit(
        X_train,
        Y_train,
        validation_data=(X_test, Y_test),
        batch_size=128,
        epochs=20,
        verbose=1
    )

    # Save compiled model
    model.save("cifar10.h5")
