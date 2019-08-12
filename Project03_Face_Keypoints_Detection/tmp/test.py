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
        plt.scatter(x, y, c='r')
    plt.show()


def random_draw(root, dataset, seed=None):
    data_root = root + "\\data"
    dataset = dataset
    path = data_root + "\\" + dataset
    file = np.loadtxt(path + "\\" + "label.txt", dtype=np.str)
    seed = seed
    if seed is not None:
        np.random.seed(seed)
    while True:
        idx = np.random.randint(0, len(file) - 1)
        info = file[idx]
        img_path = path + "\\" + info[0]
        draw(img_path, info)
        yield


if __name__ == "__main__":
    root = os.getcwd()
    datasets = ['I', 'II']
    RD = random_draw(root, datasets[0])
    for i in range(10):
        next(RD)
