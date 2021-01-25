import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


def read_text_file(file):
    with open(file) as f:
        print(f.read())


def remove_folder(path):
    if os.path.isdir(path):
        shutil.rmtree(path)


def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation), '')


def vectorize_text(vectorize_layer, text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


def run():
    if os.path.isdir('aclImdb'):
        print("Dataset offline")
        dataset_dir = os.path.join(os.path.dirname(''), 'aclImdb')
    else:
        print("Download dataset")
        url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
        dataset = tf.keras.utils.get_file("aclImdb_v1.tar.gz", url,
                                          untar=True, cache_dir='.',
                                          cache_subdir='')
        dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')

    print("Dataset dirs")
    print(os.listdir(dataset_dir))

    train_dir = os.path.join(dataset_dir, 'train')
    print("Training dir")
    print(os.listdir(train_dir))

    # Read sample file
    sample_file = os.path.join(train_dir, 'pos/1181_9.txt')
    read_text_file(sample_file)

    # Remove unused folder
    remove_dir = os.path.join(train_dir, 'unsup')
    remove_folder(remove_dir)

    # Create raw dataset (train, valid, test)
    batch_size = 32
    seed = 42

    print("raw_train_ds")
    raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
        train_dir,
        batch_size=batch_size,
        validation_split=0.2,
        subset='training',
        seed=seed
    )

    for text_batch, label_batch in raw_train_ds.take(1):
        for i in range(3):
            print("Review:", text_batch.numpy()[i])
            print("Label:", label_batch.numpy()[i])

    print("Label 0 corresponds to", raw_train_ds.class_names[0])
    print("Label 1 corresponds to", raw_train_ds.class_names[1])

    print("raw_val_ds")
    raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
        train_dir,
        batch_size=batch_size,
        validation_split=0.2,
        subset='validation',
        seed=seed
    )

    test_dir = os.path.join(dataset_dir, 'test')
    print("raw_test_ds")
    raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
        test_dir,
        batch_size=batch_size,
    )

    # Standardize text vector
    max_features = 10000
    sequence_length = 250

    vectorize_layer = TextVectorization(
        standardize=custom_standardization,
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length
    )

    # Make a text-only dataset (without labels), then call adapt
    train_text = raw_train_ds.map(lambda x, y: x)
    vectorize_layer.adapt(train_text)

    # retrieve a batch (of 32 reviews and labels) from the dataset
    text_batch, label_batch = next(iter(raw_train_ds))
    first_review, first_label = text_batch[0], label_batch[0]
    print("Review:", first_review)
    print("Label:", first_label)
    print("Vectorized review:", vectorize_text(vectorize_layer, first_review, first_label))

    print("1287 ---> ", vectorize_layer.get_vocabulary()[1287])
    print("3529 ---> ", vectorize_layer.get_vocabulary()[3529])
    print(" 313 ---> ", vectorize_layer.get_vocabulary()[313])
    print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))

    # Create real dataset
    train_ds = raw_train_ds.map(lambda x, y: vectorize_text(vectorize_layer, x, y))
    valid_ds = raw_val_ds.map(lambda x, y: vectorize_text(vectorize_layer, x, y))
    test_ds = raw_test_ds.map(lambda x, y: vectorize_text(vectorize_layer, x, y))

    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    valid_ds = valid_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    embedding_dim = 16
    model = tf.keras.Sequential([
        layers.Embedding(max_features + 1, embedding_dim),
        layers.Dropout(0.2),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])

    model.summary()

    # Add loss func
    model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
                  optimizer='adam',
                  metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

    # Training model
    epochs = 10
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=epochs
    )

    loss, accuracy = model.evaluate(test_ds)
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)

    history_dict = history.history

    print(history_dict.keys())

    acc = history_dict['binary_accuracy']
    val_acc = history_dict['val_binary_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    # Show graph of training and validation loss
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validate loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # plt.show()

    # Show graph
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    # plt.show()

    # Export model
    export_model = tf.keras.Sequential([
        vectorize_layer,
        model,
        layers.Activation('sigmoid')
    ])

    export_model.compile(
        loss=losses.BinaryCrossentropy(from_logits=False), optimizer='adam', metrics=['accuracy']
    )

    # Test it with `raw_test_ds`, which yields raw strings
    loss, accuracy = export_model.evaluate(raw_test_ds)
    print(accuracy)
    print(loss)

    examples = [
        "The movie was great!",
        "The movie was okay.",
        "The movie was terrible..."
    ]

    predict_result = export_model.predict(examples)
    print("Predict: ")
    print(predict_result)
