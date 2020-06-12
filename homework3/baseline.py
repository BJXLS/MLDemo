import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
import cv2

import load_data
np.set_printoptions(threshold=np.inf)

# cifar10 = tf.keras.datasets.cifar10
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0

# 导入数据集
data, label = load_data.load_data()
print(len(data))
# 使用相同的seed，保证输入特征和标签一一对应
np.random.seed(116)
np.random.shuffle(data)
np.random.seed(116)
np.random.shuffle(label)
tf.random.set_seed(116)

# 取五分之一作为测试
x_train = data[:-185]
y_train = label[:-185]
x_test = data[-185:]
y_test = label[-185:]

x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
x_train, x_test = x_train / 255.0, x_test / 255.0


# x_train = []
# x_test = []
# for i in x_train_:
#     x_train.append(i.reshape((-1, 227, 227, 1)))
# # x_train = x_train.reshape((-1, 277, 277, 1))
# # x_test = x_test.reshape((-1, 277, 277, 1))
#
# # print(len(x_test_))
# # print(x_test_[0])
# for j in x_test_:
#     x_test.extend(j.reshape((-1, 227, 227, 1)))


x_train = x_train.reshape((-1, 227, 227, 1))
x_test = x_test.reshape((-1, 227, 227, 1))

# from_tensor_slices函数使输入特征和标签值一一对应。（把数据集分批次，每个批次batch组数据）
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


class Baseline(Model):
    def __init__(self):
        super(Baseline, self).__init__()
        self.c1 = Conv2D(filters=6, kernel_size=(5, 5), padding="same")
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding="same")
        self.d1 = Dropout(0.2)
        self.flatten = Flatten()
        self.f1 = Dense(128, activation="relu")
        self.d2 = Dropout(0.2)
        self.f2 = Dense(10, activation="softmax")

    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)
        x = self.d1(x)
        x = self.flatten(x)
        x = self.f1(x)
        x = self.d2(x)
        y = self.f2(x)
        return y


class AlexNet(Model):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.c1 = Conv2D(filters=96, kernel_size=(3, 3))
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.p1 = MaxPool2D(pool_size=(3, 3), strides=2)

        self.c2 = Conv2D(filters=256, kernel_size=(3, 3))
        self.b2 = BatchNormalization()
        self.a2 = Activation('relu')
        self.p2 = MaxPool2D(pool_size=(3, 3), strides=2)

        self.c3 = Conv2D(filters=384, kernel_size=(3, 3), padding='same',
                         activation='relu')

        self.c4 = Conv2D(filters=384, kernel_size=(3, 3), padding='same',
                         activation='relu')

        self.c5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same',
                         activation='relu')
        self.p3 = MaxPool2D(pool_size=(3, 3), strides=2)

        self.flatten = Flatten()
        self.f1 = Dense(2048, activation='relu')
        self.d1 = Dropout(0.5)
        self.f2 = Dense(2048, activation='relu')
        self.d2 = Dropout(0.5)
        self.f3 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)

        x = self.c2(x)
        x = self.b2(x)
        x = self.a2(x)
        x = self.p2(x)

        x = self.c3(x)

        x = self.c4(x)

        x = self.c5(x)
        x = self.p3(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.d1(x)
        x = self.f2(x)
        x = self.d2(x)
        y = self.f3(x)
        return y


class AlexNetBreast(Model):
    def __init__(self):
        super(AlexNetBreast, self).__init__()
        self.c1 = Conv2D(filters=96, kernel_size=(11, 11), strides=4, padding='same')
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.p1 = MaxPool2D(pool_size=(3, 3), strides=2)

        self.c2 = Conv2D(filters=256, kernel_size=(5, 5), padding='same')
        self.b2 = BatchNormalization()
        self.a2 = Activation('relu')
        self.p2 = MaxPool2D(pool_size=(3, 3), strides=2)
        # 13*13

        self.c3 = Conv2D(filters=384, kernel_size=(3, 3), padding='same',
                         activation='relu')

        self.c4 = Conv2D(filters=384, kernel_size=(3, 3), padding='same',
                         activation='relu')

        self.c5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same',
                         activation='relu')
        self.p3 = MaxPool2D(pool_size=(3, 3), strides=2)

        self.flatten = Flatten()
        self.f1 = Dense(4096, activation='relu')
        self.d1 = Dropout(0.5)
        self.f2 = Dense(4096, activation='relu')
        self.d2 = Dropout(0.5)
        self.f3 = Dense(4, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)

        x = self.c2(x)
        x = self.b2(x)
        x = self.a2(x)
        x = self.p2(x)

        x = self.c3(x)

        x = self.c4(x)

        x = self.c5(x)
        x = self.p3(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.d1(x)
        x = self.f2(x)
        x = self.d2(x)
        y = self.f3(x)
        return y



# model = Baseline()


model = AlexNetBreast()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])
checkpoint_save_path = './checkpoint/AlexNetBreast.ckpt'
if os.path.exists(checkpoint_save_path + ".index"):
    print("----------------------load the model--------------------")
    model.load_weights(checkpoint_save_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)
history = model.fit(train_db, epochs=7, validation_data=test_db, validation_freq=1,
                    callbacks=[cp_callback])

model.summary()
print("save start...")
# file = open("./weights.txt", "w")
# for v in model.trainable_variables:
#     file.write(str(v.name) + '\n')
#     file.write(str(v.shape) + '\n')
#     file.write(str(v.numpy()) + '\n')
# file.close()
print("part1 finish...")
# ====================绘画=======================
# acc = history.history['sparse_categorical_accuracy']
# val_acc = history.history['val_sparse_categorical_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# plt.subplot(1, 2, 1)
# plt.plot(acc, label='Training Accuracy')
# plt.plot(val_acc, label='Validation Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.legend()
# plt.subplot(1, 2, 2)
# plt.plot(loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.title("Training and Validation Loss")
# plt.legend()
# plt.show()
print("part2 finsih...")

fold_path = 'new_test_cut/'
fold_list = os.listdir(fold_path)
# 遍历文件夹
for fold in fold_list:
    img_path = os.path.join(fold_path, fold)
    img_list = os.listdir(img_path)
    # 遍历文件图像
    for img_ in img_list:
        img__ = os.path.join(img_path, img_)


        # im = Image.open(img__)
        # im = im.resize((227, 227))
        # im = np.array(im)
        # im = im/255.0
        # im.reshape((-1, 227, 227, 1))

        img = cv2.imread(img__, 0)
        img = cv2.resize(img, (227, 227))
        img = np.array(img)
        img = img / 255.0
        img = img.reshape((-1, 227, 227, 1))

        ret = model.predict(img)
        file = open("./result.txt", "a+")
        file.write(str(fold) + " " + str(ret) + '\n')
        file.close()
        print(str(fold) + " " + str(ret))

print("finish")