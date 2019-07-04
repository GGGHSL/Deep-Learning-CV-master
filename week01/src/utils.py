import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt


class DataAugmentation:
    def __init__(self, file_path):
        self.file_path = file_path
        self.img = cv2.imread(file_path)

    def show_all_augmentations(self, save_root=None, dpi=3000):
        # all augmentations in default
        images = [self.random_crop(), self.random_light_color(), self.adjust_gamma(gamma=1.5),
                  self.similarity_transform(), self.affine_transform(), self.random_warp()]
        titles = ['random cropped', 'random colored', 'adjusted gamma',
                  'similarity transformed', 'affine transformed', 'random warpped']
        for (index, cv2_img_x, title) in zip(range(231, 237), images, titles):
            img_x = cv2.cvtColor(cv2_img_x, cv2.COLOR_BGR2RGB)
            plt.subplot(index)
            plt.title(title)
            plt.imshow(img_x)
        file_name = os.path.basename(self.file_path)
        new_name = file_name[:-4] + '_aug' + file_name[-4:]
        if save_root is None:
            save_root = os.path.join(os.path.dirname(self.file_path), 'show_aug_data/')
        if not os.path.exists(save_root):
            os.mkdir(save_root)
        plt.savefig(os.path.join(save_root, new_name), dpi=dpi)

    def random_augmentation(self, num=10, save_root=None, dpi=3000):
        func_list = [self.random_crop, self.random_light_color, self.adjust_gamma,
                     self.similarity_transform, self.affine_transform, self.random_warp]
        for _ in range(num):
            do = random.choice(func_list)
            cv2_img_x = do()
            img_x = cv2.cvtColor(cv2_img_x, cv2.COLOR_BGR2RGB)
            if save_root is None:
                save_root = os.path.join(os.path.dirname(self.file_path), 'augment_data/')
            if not os.path.exists(save_root):
                os.mkdir(save_root)
            file_name = os.path.basename(self.file_path)
            new_name = file_name[:-4] + "_" + str(do).split(' ', 4)[2].split('.')[-1] + file_name[-4:]
            print(new_name)
            plt.imshow(img_x)
            plt.savefig(os.path.join(save_root, new_name), dpi=dpi)

    def random_crop(self):
        img = self.img
        h, w, _ = img.shape
        r1, r2 = sorted([random.randint(0, h - 1) for _ in range(2)])
        c1, c2 = sorted([random.randint(0, w - 1) for _ in range(2)])
        img_crop = img[r1:r2, c1:c2]
        return img_crop

    def random_light_color(self, seed=None):
        """
        Add a random int between (-50, 50) to each of the three limited channels.
        ---------------------------------------------
        Parameters
          img: a original image in cv2 style (BGR)
          return: merged image (BGR)
        """
        img = self.img
        B, G, R = cv2.split(img)
        if seed is not None:
            random.seed(seed)
        b_rand, g_rand, r_rand = [random.randint(-50, 50) for _ in range(0, 3)]
        for (c, C) in zip([b_rand, g_rand, r_rand], [B, G, R]):
            if c == 0:
                pass
            elif c > 0:
                lim = 255 - c
                C[C > lim] = 255  # white: make it empty
                C[C <= lim] = (c + C[C <= lim]).astype(img.dtype)
            else:  # c < 0
                lim = 0 - c
                C[C < lim] = 0  # black
                C[C > lim] = (c + C[C > lim]).astype(img.dtype)
        img_merge = cv2.merge((B, G, R))
        return img_merge

    def adjust_gamma(self, gamma=2.):
        """
        Adjust a dark image to the proper gray level.
        ---------------------------------------------
        Parameters
          img: a original image in cv2 style (BGR)
          return: corrected image (BGR)
        """
        img = self.img
        inv_gamma = 1. / gamma
        table = np.array([255. * (i / 255) ** inv_gamma for i in range(256)]).astype("uint8")
        return cv2.LUT(img, table)

    def similarity_transform(self, new_center=None, angle=30, scale=0.6, output_size=None):
        img = self.img
        rows, cols, chs = img.shape
        if new_center is None:
            new_center = (cols / 2, rows / 2)
        if output_size is None:
            output_size = (cols, rows)
        M_srt = cv2.getRotationMatrix2D(center=new_center, angle=angle, scale=scale)
        img_srt = cv2.warpAffine(src=img, M=M_srt, dsize=output_size)
        return img_srt

    def affine_transform(self, output_size=None):
        img = self.img
        rows, cols, chs = img.shape
        if output_size is None:
            output_size = (cols, rows)
        pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
        pts2 = np.float32([[cols * 0.2, rows * 0.1],
                           [cols * 0.9, rows * 0.2],
                           [cols * 0.1, rows * 0.9]])
        # Calculates an affine transform from three pairs of the corresponding points.
        M_affine = cv2.getAffineTransform(src=pts1, dst=pts2)
        img_affine = cv2.warpAffine(src=img, M=M_affine, dsize=output_size)
        return img_affine

    def random_warp(self):
        # perspective transform
        img = self.img
        height, width, channels = img.shape
        random_margin = 60
        # src
        x1 = random.randint(-random_margin, random_margin)
        y1 = random.randint(-random_margin, random_margin)
        x2 = random.randint(width - random_margin - 1, width - 1)
        y2 = random.randint(-random_margin, random_margin)
        x3 = random.randint(width - random_margin - 1, width - 1)
        y3 = random.randint(height - random_margin - 1, height - 1)
        x4 = random.randint(-random_margin, random_margin)
        y4 = random.randint(height - random_margin - 1, height - 1)
        # dst
        dx1 = random.randint(-random_margin, random_margin)
        dy1 = random.randint(-random_margin, random_margin)
        dx2 = random.randint(width - random_margin - 1, width - 1)
        dy2 = random.randint(-random_margin, random_margin)
        dx3 = random.randint(width - random_margin - 1, width - 1)
        dy3 = random.randint(height - random_margin - 1, height - 1)
        dx4 = random.randint(-random_margin, random_margin)
        dy4 = random.randint(height - random_margin - 1, height - 1)
        # warp:
        pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
        pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
        M_warp = cv2.getPerspectiveTransform(src=pts1, dst=pts2)
        img_warp = cv2.warpPerspective(src=img, M=M_warp, dsize=(width, height))
        return img_warp