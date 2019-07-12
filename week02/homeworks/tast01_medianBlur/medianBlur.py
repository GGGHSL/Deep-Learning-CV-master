import numpy as np


def medianBlur(img, radius, padding_way):
    if isinstance(img, list):
        img = np.array(img, dtype=np.uint8)
    MB = MedianBlur(image=img, r=radius, padding_way=padding_way)
    MB.get_median_image()
    return MB.median_blur_image

class MedianBlur():
    """
    Median Blurring in O(1) Runtime Complexity.
    """
    def __init__(self, image, r, padding_way):
        self.image = image
        self.r = r
        self.padding_way = padding_way
        self.H, self.W = image.shape
        self.img = self.get_padding_img()
        self.median_blur_image = np.zeros(image.shape, dtype=np.uint8)
        """
        初始化第一行:
        """
        """ (1) 创建空矩阵: """
        # 记录包括padding在内的每一列;
        # 统计的每个小column只有2*r+1维, 为了可加, 同一到256维:
        self.h_hist = np.zeros((256, self.W+2*r), dtype=np.uint8)
        # 记录当前kernel所在位置的每一列:
        self.H_hist = np.zeros((256, 2*r+1), dtype=np.uint8)
        """ (2) 初始化h_hist: """
        for j in range(self.W + 2 * r):
            self.h_hist[:, j] = self.get_hist(j)

    def get_median_image(self):
        r = self.r
        # (i, j) is the position of current central pixel in image.
        for i in range(self.H):
            for j in range(self.W):
                i_img = i + r  # current i corresponding to i+r in padded-image img
                j_h = j + r  # current j corresponding to j+r in h_hist
                """ 1.Update h_hist and H_hist: """
                if i == 0:  # first line processed separately
                    self.H_hist = self.h_hist[:, j_h - r: j_h + r + 1]
                else:
                    if j == 0:  # first element of each line processed separately
                        j_move_down = list(range(j_h - r, j_h + r + 1))  # (2*r+1) columns in total.
                        self.move_down(i_img, j_move_down)  # first (2*r+1) columns of h_hist updated.
                        self.H_hist = self.h_hist[:, j_h - r: j_h + r + 1]  # H_hist updated.
                    else:
                        j_move_down = j_h + r
                        self.move_down(i_img, j_move_down)  # next one column in h_hist updated.
                        self.move_right(j_h)  # H_hist updated.

                """ 2.Use updated H_hist to get median value to update median_image: """
                self.median_blur_image[i, j] = self.get_median()

    # def get_median_image(self):
    #     self.median_blur_image = np.array([
    #         list(map(lambda i:
    #                  list(map(lambda j: self.get_one_median_value(i, j), range(self.W))),
    #                  range(self.H)))
    #     ], dtype=np.uint8)
    #
    # def get_one_median_value(self, i, j):
    #     # (i, j) is the position of current central pixel in image.
    #     r = self.r
    #     i_img = i + r  # current i corresponding to i+r in padded-image img
    #     j_h = j + r  # current j corresponding to j+r in h_hist
    #     """ 1.Update h_hist and H_hist: """
    #     if i == 0:  # first line processed separately
    #         self.H_hist = self.h_hist[:, j_h - r: j_h + r + 1]
    #     else:
    #         if j == 0:  # first element of each line processed separately
    #             j_move_down = list(range(j_h - r, j_h + r + 1))  # (2*r+1) columns in total.
    #             self.move_down(i_img, j_move_down)  # first (2*r+1) columns of h_hist updated.
    #             self.H_hist = self.h_hist[:, j_h - r: j_h + r + 1]  # H_hist updated.
    #         else:
    #             j_move_down = j_h + r
    #             self.move_down(i_img, j_move_down)  # next one column in h_hist updated.
    #             self.move_right(j_h)  # H_hist updated.
    #     """ 2.Use updated H_hist to get median value to update median_image: """
    #     return self.get_median()

    """
    0. 定义padding函数
    """
    def get_padding_img(self):
        assert self.padding_way in ["REPLICA", "ZERO"]
        if self.padding_way == "ZERO":
            img = np.pad(self.image, pad_width=self.r, mode='constant', constant_values=0)
        else:
            img = np.pad(self.image, pad_width=self.r, mode='edge')
        return img

    """
    1. 定义get_hist函数
    """
    def get_hist(self, j):
        col = self.img[0:(2 * self.r + 1), j]
        hist = np.zeros(256, dtype=np.uint8)
        for val in col:
            hist[val] += 1

        return hist

    """
    2. 定义位移函数: move_down(), move_right()
    """
    # First！
    # update h_hist:
    def move_down(self, i, j_move):
        # i is the line of current central pixel in padded-image img;
        # j_move is the list of columns to be moved in h_hist.
        i_remove = i - self.r - 1
        i_add = i + self.r
        if isinstance(j_move, int):
            j_move = [j_move, ]
        for j in j_move:
            remove_val = self.img[i_remove, j]
            add_val = self.img[i_add, j]
            self.h_hist[remove_val, j] -= 1
            self.h_hist[add_val, j] += 1

    # Second!
    # update H_hist:
    def move_right(self, j_h):
        # j_h is the column of current central pixel in h_hist.
        # 注意: 此时的H_hist尚未更新,还处于上一个位置;但j已经更新,是当前位置的列.
        # j_remove = j_h - r - 1是相对于h_hist来说的,事实上H_hist永远只需要del第0列.
        j_add = j_h + self.r
        self.H_hist = np.c_[np.delete(self.H_hist, 0, axis=1),
                            self.h_hist[:, j_add]]

    """
    3. 定义求中值函数: get_median()
    """
    def get_median(self):
        # get median value for current H_hist
        hist = np.sum(self.H_hist, axis=1)
        thres = (2 * self.r + 1) ** 2 // 2 + 1
        sum_cnt = 0
        median = 0
        for val in range(256):
            cnt = hist[val]
            sum_cnt += cnt
            if sum_cnt >= thres:
                median = val
                break
        return median
