# This code came from Open AIs Chat GPT and is being used by Christopher Bruinsma and Syed
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.ops.numpy_ops import np_config
import matplotlib.pyplot as plt
from progress.bar import Bar # Style — literally just style
import cv2 as photo

# Reduced batch size to lower the CPU load and to accelerate the processing
image_size = (2000, 2000)
input_shape=(2000, 2000, 3)
batch_size = 15



import tensorflow as tf


# Define the generator network
def make_generator_model():
    model = keras.Sequential()
    model.add(layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.PReLU())
    model.add(layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.PReLU())
    model.add(layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.PReLU())
    model.add(layers.Conv2DTranspose(3, (3, 3), strides=(1, 1), padding='same', activation='tanh'))
    model.add(layers.Dropout(0.3))
    return model

# Define the discriminator network
def make_discriminator_model():
    model = keras.Sequential()
    model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=input_shape))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='relu'))

    return model

# Define the loss functions
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
mse_loss = tf.keras.losses.MeanSquaredError()

# Define the optimizer
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Define the training loop
@tf.function
def train_step(images):
    # Add random noise to the input images
    # # Define the ground truth labels
    real_labels = tf.ones((images.shape[0], 1))
    fake_labels = tf.zeros((images.shape[0], 1))

    noisy_images = images

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generate denoised images
        generated_images = generator(noisy_images, training=True)

        # Calculate generator loss
        gen_loss = gen_loss = mse_loss(images, generated_images)

        # Calculate discriminator loss for real images
        real_output = discriminator(generated_images, training=True)
        disc_real_loss = cross_entropy(real_labels, real_output)

        # Calculate discriminator loss for fake images
        fake_output = discriminator(generated_images, training=True)
        disc_fake_loss = cross_entropy(fake_labels, fake_output)

        # Calculate total discriminator loss
        disc_loss = disc_real_loss + disc_fake_loss

    # Calculate gradients and apply to the generator and discriminator variables
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

# Create the generator and discriminator models
generator = make_generator_model()
discriminator = make_discriminator_model()

# Define the training parameters
EPOCHS = 5
BATCH_SIZE = 3

# # Define the training loop
# train_images = tfds.as_numpy(train_images)
# print(train_images)

# Load the dataset and preprocess the images
(train_images, _), (test_images, _) = tf.keras.utils.image_dataset_from_directory(
    directory="Training_Data",
    labels='inferred',
    class_names=None,
    color_mode='rgb',
    shuffle=True,
    batch_size=32,
    image_size=image_size,
    crop_to_aspect_ratio=True,
)

# Display the model's architecture
generator.summary()


def train():
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        for batch in range(0, train_images.shape[0], BATCH_SIZE):
            # Get the next batch of images
            batch_images = train_images[batch:batch+BATCH_SIZE]

            # Train the generator and discriminator on the batch
            gen_loss, disc_loss = train_step(batch_images)

            # Print the progress
            print(f"Batch {batch+1}/{train_images.shape[0]}: Generator Loss={gen_loss:.4f}, Discriminator Loss={disc_loss:.4f}")

    # Evaluate the model on the test set
    test_loss = mse_loss(test_images, generator(test_images, training=False)).numpy()
    print(f"Test Loss: {test_loss:.4f}")

train()

noisy_images = test_images[:5]

path = "training_2/cp-{epoch:04d}.ckpt".format(epoch = EPOCHS)
generator.save_weights(path)


# Generate denoised images using the trained generator model

def MakeModelPlots(generator,discriminator):
    tf.keras.utils.plot_model(
        generator,
        to_file='generator.png',
        show_shapes=True,
        show_dtype=False,
        show_layer_names=True,
        rankdir='TB',
        expand_nested=False,
        dpi=96,
        layer_range=None,
        show_layer_activations=False,
    )

    tf.keras.utils.plot_model(
        discriminator,
        to_file='discriminator.png',
        show_shapes=False,
        show_dtype=True,
        show_layer_names=True,
        rankdir='TB',
        expand_nested=False,
        dpi=96,
        layer_range=None,
        show_layer_activations=False,
    )

def plot_generator(generator):
    tf.keras.utils.plot_model(
        generator,
        to_file='generator.png',
        show_shapes=True,
        show_dtype=False,
        show_layer_names=True,
        rankdir='TB',
        expand_nested=False,
        dpi=96,
        layer_range=None,
        show_layer_activations=False,
    )
# Loads the weights
generator.load_weights(path)
generator.summary()
generator.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
 
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
tf.keras.utils.plot_model(
        generator,
        to_file='generator.png',
        show_shapes=True,
        show_dtype=False,
        show_layer_names=True,
        rankdir='TB',
        expand_nested=False,
        dpi=96,
        layer_range=None,
        show_layer_activations=False,
)

plot_generator(generator)
generated_images = generator(noisy_images, training=False)

# Display the original noisy images and the generated denoised images
for i in range(5):
    plt.imshow(generated_images[i])
    plt.axis('off')
    plt.show()



