from PIL import Image
import numpy as np
import os

# model = baseline.AlexNetBreast()
# checkpoint_save_path = './checkpoint/AlexNetBreast.ckpt'
# model.load_weights(checkpoint_save_path)

fold_path = 'new_test_cut/'
fold_list = os.listdir(fold_path)
# 遍历文件夹
for fold in fold_list:
    img_path = os.path.join(fold_path, fold)
    img_list = os.listdir(img_path)
    # 遍历文件图像
    for img_ in img_list:
        img__ = os.path.join(img_path, img_)
        im = Image.open(img__)
        im = im.resize((227, 227))
        im = np.array(im)
        (im/255.0).reshape((-1, 227, 227, 1))
        # ret = model.predict((im).reshape((1, 227, 227)))
        file = open("./result.txt", "a+")
        # file.write(str(fold) + " " + str(ret) + '\n')
        file.close()
        # print(str(fold) + " " + str(ret))

print("finish")

