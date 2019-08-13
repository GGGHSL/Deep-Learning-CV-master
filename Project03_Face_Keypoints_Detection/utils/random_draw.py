import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def draw(img_path, info):
    img = Image.open(img_path)
    img_name = info[0]
    print(img_name)
    bbox = info[1:5].astype(np.float32)
    x1, y1 = bbox[0:2]  # 左上角
    x2, y2 = bbox[2:4]  # 右下角
    keypoints = info[5:47].astype(np.float32).reshape(21, 2)
    plt.figure()
    plt.imshow(img)
    plt.title(img_name)
    plt.gca().add_patch(  # 获取当前子图plt.gca(): Get Current Axes
        plt.Rectangle(xy=(x1, y1), width=x2 - x1, height=y2 - y1,
                      fill=False, edgecolor='g', linewidth=3))
    for x, y in keypoints:
        plt.gca().scatter(x, y, c='r', s=10)
    plt.show()


def draw_crop(img_path, info):
    img = Image.open(img_path)
    img_name = info[0]
    print(img_name)
    x1, y1, x2, y2 = info[1:5].astype(np.float32)
    roi = (x1, y1, x2, y2)
    crop = img.crop(roi)
    keypoints = info[5:47].astype(np.float32).reshape(21, 2)
    plt.figure()
    plt.imshow(crop)
    plt.title(img_name)
    for x, y in keypoints:
        plt.gca().scatter(x, y, c='r', s=10)
    plt.show()


def random_draw(file_path, crop=False, shuffle=True, seed=None):
    """
    随机选取一张图片显示, 画出相应人脸边框和关键点.
    :param file_path: 标注信息文件地址
    :param crop: bool类型, 是否只画出用人脸边框截取的部分
    :param shuffle: 图片是否随机选取, 否则按标注信息顺序显示
    :param seed: 设定随机种子
    :return: 函数迭代器
    """
    file = np.loadtxt(file_path, dtype=np.str)
    idx = -1
    if seed is not None:
        np.random.seed(seed)
    while True:
        if shuffle:
            idx = np.random.randint(0, len(file) - 1)
        else:
            idx += 1
        info = file[idx]
        img_path = root + "\\data\\" + "I\\" + info[0]
        if not os.path.exists(img_path):
            img_path = root + "\\data\\" + "II\\" + info[0]
        if os.path.exists(img_path):
            if not crop:
                draw(img_path, info)
            else:
                draw_crop(img_path, info)
        else:
            print("Image {} does not exist!".format(info[0]))
        yield


if __name__ == "__main__":
    root = os.getcwd()
    print(root)
    RD = random_draw(root)
    for i in range(10):
        next(RD)
