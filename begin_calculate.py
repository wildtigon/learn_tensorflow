import tensorflow as tf;
import matplotlib.pyplot as plt


def run():
    print(tf.__version__)
    print(tf.reduce_sum(tf.random.normal([1000, 1000])))


def sample_load_data():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    print(train_images.shape)
    print(len(train_images))


def sample_show_image():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    plt.figure()
    plt.imshow(train_images[0])
    plt.colorbar()
    plt.grid(True)
    plt.show()