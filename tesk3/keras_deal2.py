from keras import models
from keras import layers
from keras.utils import to_categorical
import numpy as np
from keras.utils.vis_utils import plot_model
"""
from keras import optimizers
from keras import applications
from keras.models import Model
from keras.layers import Dense, Input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from matplotlib import pyplot
import matplotlib.pyplot as plt
"""

con_idx = ((np.asarray([i + j for i in range(0, 600, 150) for j in range(15)])[
                (np.array([7, 8, 9]).reshape(1, -1) * np.ones((50, 3))).reshape(5, 30).astype('int32')] + (
                    (np.arange(10) * 15).reshape(-1, 1) * np.ones(3)).reshape(1, -1)) + (
                   (np.arange(5) * 600).reshape(-1, 1) * np.ones(30))).ravel().astype('int32')
"""
datagen = ImageDataGenerator(rotation_range=90, width_shift_range=0.5, height_shift_range=0.5,
                             brightness_range=[0.1, 10], zoom_range=[0.1, 1],
                             horizontal_flip=True, vertical_flip=True,
                             featurewise_center=True, featurewise_std_normalization=True)
"""

# datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)

all_date = np.load('all_date.npy')

x_train = all_date[np.array([i for i in range(3000) if i not in con_idx])].reshape(-1, 28, 28, 1)
x_test = all_date[con_idx].reshape(-1, 28, 28, 1)

y_train = (np.arange(5).reshape(-1, 1) * np.ones((5, 570))).ravel().astype('uint8')
y_test = (np.arange(5).reshape(-1, 1) * np.ones((5, 30))).ravel().astype('uint8')

# datagen.fit(x_train)

idx = np.arange(x_train.shape[0])
np.random.shuffle(idx)
x_train = x_train[idx]
y_train = y_train[idx]

idx2 = np.arange(x_test.shape[0])
np.random.shuffle(idx2)
x_test = x_test[idx2]
y_test = y_test[idx2]

train_labels = to_categorical(y_train)
y_test = to_categorical(y_test)
"""
base_model = applications.VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(28, 28, 1)), classes=5)

for layer in base_model.layers:
    print(layer.trainable)
    layer.trainable = False
"""
##############################################################

##############################################################
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
# model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(5, activation='softmax'))
######################################################################

######################################################################
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, train_labels, epochs=50, batch_size=128)

test_loss, test_acc = model.evaluate(x_test, y_test)
print('acc: ', test_acc)

model.save("./save/model_6.ckpt")
plot_model(model, to_file='model.png', show_shapes=True)

"""
for ba_data, ba_labels in datagen.flow(x_train, train_labels, batch_size=9):
    for i in range(0, 9):
        pyplot.subplot(330 + 1 + i)
        pyplot.imshow(ba_data[i].reshape(28, 28), cmap=pyplot.get_cmap('gray'))
    pyplot.show()
    break
"""
