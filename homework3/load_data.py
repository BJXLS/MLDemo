import cv2
import os
import numpy as np
import tensorflow as tf


def create_model():
    pass

def load_data():
    img_path_ = 'cut_img/'

    data = []
    label = []
    a = 0
    # 遍历1-4 四个文件夹
    for i in range(1, 5):
        fold_path = img_path_ + str(i)
        fold_list = os.listdir(fold_path)

        # 遍历每个label文件夹里的文件夹
        for j in range(0, len(fold_list)):
            temp_path = os.path.join(fold_path, fold_list[j])
            img_list = os.listdir(temp_path)

            # 遍历每个文件夹里的图片
            a += len(img_list)
            print(a)

            for k in range(0, len(img_list)):
                img_path = os.path.join(temp_path, img_list[k])
                # 读入灰度图像
                img = cv2.imread(img_path, 0)
                img = cv2.resize(img, (227, 227))
                # 把图片加入data中
                data.append(img)
                # 记录label
                label.append(i-1)
        print(i)
    print(a)
    print("finish")
    return data, label

if __name__ == '__main__':
    data, label = load_data()
    np.random.seed(116)  # 使用相同的seed，保证输入特征和标签一一对应
    np.random.shuffle(data)
    np.random.seed(116)
    np.random.shuffle(label)
    tf.random.set_seed(116)

    # 将打乱后的数据集分割为训练集和测试集，训练集为前120行，测试集为后30行
    x_train = data[:-30]
    y_train = data[:-30]
    x_test = label[-30:]
    y_test = label[-30:]

    # 转换x的数据类型，否则后面矩阵相乘时会因数据类型不一致报错
    x_train = tf.cast(x_train, tf.float32)
    x_test = tf.cast(x_test, tf.float32)

    # from_tensor_slices函数使输入特征和标签值一一对应。（把数据集分批次，每个批次batch组数据）
    train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)