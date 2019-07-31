import sys
import cv2
import os
from matplotlib import pyplot as plt
sys.path.append('../task01_medianBlur')
from medianBlur import *
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    image_address = '../20190712182540.jpg'
    image = cv2.imread(image_address)
    img = cv2.resize(image, dsize=(int(image.shape[1] / 2), int(image.shape[0] / 2)))
    H, W, C = img.shape
    B, G, R = cv2.split(img)
    # Radius: r = 3
    # kernel size = 2 * r + 1 = 7
    padding_way = "REPLICA"
    # padding_ways = ["REPLICA", "ZERO"]
    r = 3
    bgr_3 = list(map(lambda _: medianBlur(_, r, padding_way), [B, G, R]))
    mb_img_3 = cv2.merge(bgr_3)

    mb_img_5 = cv2.merge(list(map(lambda _: medianBlur(_, 5, padding_way), [B, G, R])))
    mb_img_7 = cv2.merge(list(map(lambda _: medianBlur(_, 7, padding_way), [B, G, R])))

    save_path = "./result"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    plt.figure(figsize=(15, 10))
    plt.subplot(221)
    plt.title('original')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.subplot(222)
    plt.title('kernel size = 3')
    plt.imshow(cv2.cvtColor(mb_img_3, cv2.COLOR_BGR2RGB))
    plt.subplot(223)
    plt.title('kernel size = 5')
    plt.imshow(cv2.cvtColor(mb_img_5, cv2.COLOR_BGR2RGB))
    plt.subplot(224)
    plt.title('kernel size = 7')
    plt.imshow(cv2.cvtColor(mb_img_7, cv2.COLOR_BGR2RGB))
    plt.savefig(save_path+"/medianBlur_kernels.jpg", dpi=1000)
    plt.show()
