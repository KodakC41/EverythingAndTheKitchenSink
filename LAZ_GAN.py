# This is an implementation of LAZ GAN for the Lazarus Lab by Christopher Bruinsma and Syed Shihan. 
import os
import numpy as np
from scipy import linalg
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.ops.numpy_ops import np_config
import matplotlib.pyplot as plt
from progress.bar import Bar # Style — literally just style
import cv2 as photo
from multiprocessing import Process


# Reduced batch size to lower the CPU load and to accelerate the processing
image_size = (1600, 1600)
input_shape=( 1600, 1600, 3)
batch_size = 15


# Define the generator network
def make_generator_model():
    model = keras.Sequential()
    model.add(layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same', input_shape=input_shape,activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.1))
    model.add(layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(3, (3, 3), strides=(1, 1), padding='same', activation='tanh'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))
    return model

# Define the discriminator network
def make_discriminator_model():
   model = keras.Sequential()
   model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=input_shape))
   model.add(layers.LeakyReLU())
   model.add(layers.Dropout(0.2))
   model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
   model.add(layers.LeakyReLU())
   model.add(layers.Dropout(0.3))
   model.add(layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
   model.add(layers.LeakyReLU())
   model.add(layers.Dropout(0.3))
   model.add(layers.Flatten())
   model.add(layers.Dense(1, activation='softmax'))
   return model

# Define the loss functions
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
mse_loss = tf.keras.losses.MeanSquaredError()

# Define the optimizer
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Define the training loop
@tf.function
def train_step(images,real_images):
    # Add random noise to the input images
    # # Define the ground truth labels
    real_labels = tf.ones((images.shape[0], 1))
    fake_labels = tf.zeros((images.shape[0], 1))

    noisy_images = images

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generate denoised images
        generated_images = generator(noisy_images, training=True)

        # Calculate generator loss
        gen_loss = gen_loss = mse_loss(real_images, generated_images)

        # Calculate discriminator loss for real images
        real_output = discriminator(real_images, training=True)
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
EPOCHS = 20
BATCH_SIZE = 10


"""
Convert images to rbg
"""
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

# Load the RAW Images
(train_images, _), (test_images, _) = tf.keras.utils.image_dataset_from_directory(
    directory="Training_Data",
    labels='inferred',
    class_names=None,
    color_mode='rgb',
    batch_size=23,
    shuffle=False,
    image_size=image_size,
    crop_to_aspect_ratio=True,
)
# Load the Transformed Images
(train_images_2, _), (test_images_2, _) = tf.keras.utils.image_dataset_from_directory(
    directory="Training_Data_2",
    labels='inferred',
    class_names=None,
    color_mode='rgb',
    batch_size=23,
    shuffle=False,
    image_size=image_size,
    crop_to_aspect_ratio=True,
)

# Compile and fit the generator to the noisy images
# def fit_disc(images):
#     real_labels = tf.ones((images.shape[0], 1))
#     with tf.GradientTape() as disc_tape:
#         disc_loss = 0
#         TempBar = Bar('Fitting the Discriminator',fill='.',index=0,max=10)
#         for e in range(10):
#             # Calculate discriminator loss for real images
#             real_output = discriminator(images, training=True)
#             disc_real_loss = cross_entropy(real_labels, real_output)
#             # Calculate total discriminator loss
#             disc_loss = disc_real_loss
#             TempBar.next()
#         gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
#         discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))



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

def compile():
    generator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),loss="mse",metrics=['mae'])
    generator.fit(train_images,test_images,epochs=EPOCHS)
    path = "training_2/cp-{epoch:04d}.ckpt".format(epoch = EPOCHS)
    generator.save_weights(path)
    plot_generator(generator)

def t_rain():
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        for batch in range(0, train_images.shape[0], BATCH_SIZE):
            # Get the next batch of images
            batch_images = train_images[batch:batch+BATCH_SIZE]
            test_images = train_images_2[batch:batch+BATCH_SIZE]
            # Train the generator and discriminator on the batch
            gen_loss, disc_loss = train_step(batch_images,test_images)

            # Print the progress
            print(f"Batch {batch+1}/{train_images.shape[0]}: Generator Loss={gen_loss:.4f}, Discriminator Loss={disc_loss:.4f}")

    # Evaluate the model on the test set
    test_loss = mse_loss(test_images, generator(test_images, training=False)).numpy()
    print(f"Test Loss: {test_loss:.4f}")


"""
Cites: https://wandb.ai/ayush-thakur/gan-evaluation/reports/How-to-Evaluate-GANs-using-Frechet-Inception-Distance-FID---Vmlldzo0MTAxOTI
"""
def fid(ground_truth, transformed_image):
     ground_truth = rgb2gray(ground_truth.numpy())
     transformed_image = rgb2gray(transformed_image.numpy())
     # calculate mean and covariance statistics
     mu1, sigma1 = ground_truth.mean(axis=0), np.cov(ground_truth, rowvar=False)
     mu2, sigma2 = transformed_image.mean(axis=0), np.cov(transformed_image,  rowvar=False)
     # calculate sum squared difference between means
     ssdiff = np.sum((mu1 - mu2)**2.0)
     # calculate sqrt of product between cov
     covmean = linalg.sqrtm(sigma1.dot(sigma2))
     # check and correct imaginary numbers from sqrt
     if np.iscomplexobj(covmean):
       covmean = covmean.real
     # calculate score
     fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
     return fid





# Does the thing
compile()
# fit_disc(test_images_2)
generator.summary()
t_rain()

real_images = test_images_2[:5]
noisy_images = test_images[:5]
generated_images = generator(noisy_images, training=False)

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


# Loads the weights
# generator.load_weights(path)
# generator.compile(optimizer='adam',
#                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#                 metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
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

# plot_generator(generator)
# generated_images = generator(noisy_images, training=False)


def calc_fid():
    for i in range(5):
        print(fid(real_images[i],generated_images[i]))
calc_fid()
# # Display the original noisy images and the generated denoised images
for i in range(5):
    plt.imshow(generated_images[i])
    plt.axis('off')
    plt.show()

