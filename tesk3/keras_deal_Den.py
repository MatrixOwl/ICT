from keras import optimizers
from keras import applications
from keras.models import Sequential, Model
# from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout, Flatten, Dense
# from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import numpy as np

con_idx = ((np.asarray([i + j for i in range(0, 600, 150) for j in range(15)])[
                (np.array([0, 1, 2]).reshape(1, -1) * np.ones((50, 3))).reshape(5, 30).astype('int32')] + (
                    (np.arange(10) * 15).reshape(-1, 1) * np.ones(3)).reshape(1, -1)) + (
                   (np.arange(5) * 600).reshape(-1, 1) * np.ones(30))).ravel().astype('int32')

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

train_labels = to_categorical(y_train)
y_test = to_categorical(y_test)

base_model = applications.VGG16(weights="imagenet", include_top=False, input_shape=(-1, 28, 28, 1))
print(base_model.summary())

for layer in base_model.layers[:15]:
    layer.trainable = False

model = Sequential()
model.add(Flatten(input_shape=base_model.output_shape[1:]))
model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))
model.add(Dense(5, activation='sigmoid'))

model = Model(inputs=base_model.input, outputs=model(base_model.output))  # 新网络=预训练网络+自定义网络

model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
              metrics=['accuracy'])
"""
train_datagen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True)  # 训练数据预处理器，随机水平翻转
test_datagen = ImageDataGenerator(rescale=1. / 255)  # 测试数据预处理器
train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(img_height, img_width),
                                                    batch_size=batch_size, class_mode='binary')  # 训练数据生成器
validation_generator = test_datagen.flow_from_directory(validation_data_dir, target_size=(img_height, img_width),
                                                        batch_size=batch_size, class_mode='binary',
                                                        shuffle=False)  # 验证数据生成器
checkpointer = ModelCheckpoint(filepath='dogcatmodel.h5', verbose=1, save_best_only=True)  # 保存最优模型

# 训练&评估
model.fit_generator(train_generator, steps_per_epoch=nb_train_samples // batch_size, epochs=epochs,
                    validation_data=validation_generator, validation_steps=nb_validation_samples // batch_size,
                    verbose=2, workers=12, callbacks=[checkpointer])
"""
test_loss, test_acc = model.evaluate(x_test, y_test)
print('acc: ', test_acc)
