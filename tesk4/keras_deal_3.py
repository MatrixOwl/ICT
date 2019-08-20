from keras import models
from keras import layers
from keras.utils import to_categorical
from sklearn import model_selection as ms
import numpy as np
from keras.datasets import mnist
from keras.utils.vis_utils import plot_model
import keras
from keras.models import model_from_json

con_idx = ((np.asarray([i + j for i in range(0, 600, 150) for j in range(15)])[
                (np.array([7, 8, 9]).reshape(1, -1) * np.ones((50, 3))).reshape(5, 30).astype('int32')] + (
                    (np.arange(10) * 15).reshape(-1, 1) * np.ones(3)).reshape(1, -1)) + (
                   (np.arange(5) * 600).reshape(-1, 1) * np.ones(30))).ravel().astype('int32')

x_new = np.load('x_train.npy').reshape(-1, 28, 28, 1).astype('float32')/255
y_new = np.load('y_train.npy').astype('uint8')
x_train_n, x_test_n, y_train_n, y_test_n = ms.train_test_split(x_new, y_new, test_size=0.2, random_state=19)

all_date = np.load('all_date.npy')

x_train = all_date[np.array([i for i in range(3000) if i not in con_idx])].reshape(-1, 28, 28, 1)
x_test = all_date[con_idx].reshape(-1, 28, 28, 1)

y_train = (np.arange(5).reshape(-1, 1) * np.ones((5, 570))).ravel().astype('uint8')
y_test = (np.arange(5).reshape(-1, 1) * np.ones((5, 30))).ravel().astype('uint8')

idx = np.arange(x_train.shape[0])
np.random.shuffle(idx)
x_train = x_train[idx]
y_train = y_train[idx]

idx2 = np.arange(x_test.shape[0])
np.random.shuffle(idx2)
x_test = x_test[idx2]
y_test = y_test[idx2]

(train_images, train_labels), (test_images, _) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

train_images = np.concatenate((train_images, x_train_n))
train_labels = np.concatenate((train_labels, y_train_n))

train_labels = to_categorical(train_labels)  # train_image
# test_labels = to_categorical(test_labels)  # test_image
y_test = to_categorical(y_test)  # x_test
y_train = to_categorical(y_train)  # x_train
y_test_n = to_categorical(y_test_n)  # x_test_n

######################################################################################################

# base_model = keras.models.load_model('model_n.h5')
# base_model = keras.Model(base_model.layers[1].input, base_model.layers[-2].output)
"""
inputs = keras.Input((28, 28, 1))
base_model = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
base_model = layers.MaxPooling2D((2, 2))(base_model)
base_model = layers.Conv2D(64, (3, 3), activation='relu')(base_model)
base_model = layers.MaxPooling2D((2, 2))(base_model)

base_model = layers.Flatten()(base_model)
base_model = layers.Dense(64, activation='relu')(base_model)
base_model = layers.Dense(10, activation='softmax')(base_model)
"""
# base_model.trainable = False

base_model = models.Sequential()
base_model.add(layers.InputLayer((28, 28, 1)))
base_model.add(layers.Conv2D(32, (3, 3), activation='relu'))
base_model.add(layers.MaxPooling2D((2, 2)))
base_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
base_model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))  # ?
base_model.add(layers.Flatten())
base_model.add(layers.Dense(64, activation='relu'))
base_model.add(layers.Dense(10, activation='softmax'))

#################################################################################################

base_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
base_model.fit(train_images, train_labels, epochs=1, batch_size=64, validation_data=(x_test_n, y_test_n))

base_model.save('model_b.h5')

base_model_n = keras.models.load_model('model_b.h5')

"""
h_file = open('model_b.h5', 'r')
load_m = h_file.read()
h_file.close()
base_model_n = keras.models.model_from_json(load_m)

base_model_n = models.Sequential()
base_model_n.load_weight('model_b.h5', by_name=True)
"""

base_model_n = keras.Model(base_model_n.layers[1].input, base_model_n.layers[-2].output)

#################################################################################################

base_model_n.trainable = False

inputs = keras.Input(shape=(28, 28, 1))
model = base_model(inputs)

model = layers.Flatten()(model)
model = layers.Dense(128, activation='relu')(model)
model = layers.Dense(64, activation='relu')(model)
model = layers.Dropout(0.2)(model)

outputs = layers.Dense(5, activation='softmax')(model)

dis_model = keras.Model(inputs=base_model.input, outputs=model(base_model.output))

#####################################################################################################

dis_model.compile(optimizer=keras.optimizers.SGD(lr=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])

dis_model.summary()

#####################################################################################################

#####################################################################################################

dis_model.fit(x_train, y_train, epochs=60, batch_size=64, validation_data=(x_test, y_test))

test_loss, test_acc = dis_model.evaluate(x_test, y_test)
print('test_acc:', test_acc)

#####################################################################################################

#####################################################################################################

dis_model.save('./save/model3.ckpt')

plot_model(dis_model, to_file='./save/model3.png', show_shapes=True)
