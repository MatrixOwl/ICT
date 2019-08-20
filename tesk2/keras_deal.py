from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
from sklearn import model_selection as ms
import numpy as np
import matplotlib.pyplot as plt
import keras


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(predicted_label,
                    100 * np.max(predictions_array), true_label), color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_labele = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.yticks(rotation=45)
    plt.xticks(range(10))
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_labele].set_color('blue')


def plot_all(num_rows, num_cols, model, x_test, y_test):
    y_num = np.where(y_test==1)[1].astype('uint8')
    num_images = num_rows * num_cols
    npredictions = model.predict(x_test)
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, npredictions, y_num, x_test.reshape(-1, 28, 28))
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, npredictions, y_num)
    plt.show()


def deal():

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images.reshape((60000, 28, 28, 1))
    train_images = train_images.astype('float32') / 255
    test_images = test_images.reshape((10000, 28, 28, 1))
    test_images = test_images.astype('float32') / 255

    x_new = np.load('x_train.npy').reshape(-1, 28, 28, 1).astype('float32')/255
    y_new = np.load('y_train.npy').astype('uint8')
    x_train, x_test, y_train, y_test = ms.train_test_split(x_new, y_new, test_size=0.2, random_state=19)

    train_images = np.concatenate((x_train, train_images))
    train_labels = np.concatenate((y_train, train_labels))

    print('ok')

    train_labels = to_categorical(train_labels)
    # test_labels = to_categorical(test_labels)
    y_test = to_categorical(y_test)
    """
    model = models.Sequential()
    model.add(layers.InputLayer((28, 28, 1)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(64, (3, 3), activation='relu'))  # ?
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    """

    inputs = keras.Input((28, 28, 1))
    base_model = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    base_model = layers.MaxPooling2D((2, 2))(base_model)
    base_model = layers.Conv2D(64, (3, 3), activation='relu')(base_model)
    base_model = layers.MaxPooling2D((2, 2))(base_model)

    base_model = layers.Flatten()(base_model)
    base_model = layers.Dense(64, activation='relu')(base_model)
    base_model = layers.Dense(10, activation='softmax')(base_model)

    base_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    base_model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_data=(x_test, y_test))

    print('train is done')

    test_loss, test_acc = base_model.evaluate(x_test, y_test)
    print('test_acc:', test_acc)
    base_model.save("./save/base_model.h5")

    plot_all(5, 5, base_model, x_test, y_test)


deal()
