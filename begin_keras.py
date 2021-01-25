import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def show_one_picture(data):
    plt.figure()
    plt.imshow(data)
    plt.colorbar()
    plt.grid(True)
    plt.show()


def show_list_pictures(images, labels, class_names, count):
    plt.figure(figsize=(10, 10))
    for i in range(count):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(True)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[labels[i]])
    plt.show()


def plot_image(i, predictions_array, true_label, class_names, img):
    true_label, img = true_label[i], img[i]
    plt.grid(True)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f} ({})".format(
        class_names[predicted_label],
        100 * np.max(predictions_array),
        class_names[true_label],
        color=color
    ))


def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(True)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim(([0, 1]))
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


def grab_model(predictions, index, test_labels, class_names, test_images):
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(index, predictions[index], test_labels, class_names, test_images)
    plt.subplot(1, 2, 2)
    plot_value_array(index, predictions[index], test_labels)
    plt.show()


def grab_all_model(predictions, test_labels, test_images, class_names):
    num_rows = 5
    num_cols = 3
    num_images = num_cols * num_rows - 1
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))

    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions[i], test_labels, class_names, test_images)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions[i], test_labels)

    plt.tight_layout()
    plt.show()


def run():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    print(train_images.shape)
    print(len(train_images))

    print(train_labels)
    print(train_images.shape)
    print(len(train_images))

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # show_one_picture(train_images[0])
    # show_list_pictures(train_images, train_labels, class_names, 25)
    # show_list_pictures(test_images, test_labels, class_names, 25)

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=10)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print('\nTest accuracy: ', test_acc)

    probability_model = tf.keras.Sequential([
        model,
        tf.keras.layers.Softmax()
    ])

    # Return all result
    predictions = probability_model(test_images)
    print("\nList predictions")
    print(predictions[0])

    # Then get the largest result
    largest_prediction = np.argmax(predictions[0])
    print("\nLargest prediction")
    print(largest_prediction)

    index = 12
    #grab_model(predictions, index, test_labels, class_names, test_images)
    grab_all_model(predictions, test_labels, test_images, class_names)

