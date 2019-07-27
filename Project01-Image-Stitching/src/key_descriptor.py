import numpy as np
import cv2
import random
from cachetools import LRUCache, cachedmethod
from operator import attrgetter


class SIFT:
    def detect_and_compute(self, img, draw=True, image_name=" "):
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(img, None)
        if draw:
            img_kp = cv2.drawKeypoints(img, kp, outImage=np.array([]),
                                       flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imshow(image_name, img_kp)
            key = cv2.waitKey()
            if key == 27:
                cv2.destroyAllWindows()
        # descriptor归一化
        sum = np.sum(des, axis=0)
        des /= sum
        return kp, des

    def get_matching_points(self, Kp1, Des1, Kp2, Des2, corrThres=0.99):
        # Keypoints matching: where the correlation matrix of descriptors is bigger than 'corrThres'
        corr = np.corrcoef(np.r_[Des1, Des2])
        corr = corr[0:Des1.shape[0], Des1.shape[0]:-1]
        loc = np.argwhere(corr > corrThres)
        index1, index2 = loc[:, 0], loc[:, 1]
        P1 = np.array([Kp1[_].pt for _ in index1])  # matched key points
        P2 = np.array([Kp2[_].pt for _ in index2])
        return P1, P2

    def draw_spindle(self, img1, img2, pt1, pt2, lwd=1, draw_samples=None):
        if not isinstance(pt1, np.ndarray):
            pt1 = np.array(pt1)
        if not isinstance(pt2, np.ndarray):
            pt2 = np.array(pt2)
        assert pt1.shape[0] == pt2.shape[0]
        n = pt1.shape[0]

        dim1, dim2 = img1.ndim, img2.ndim
        if dim1 == 2:
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        if dim2 == 2:
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        h1, w1, c1 = img1.shape
        h2, w2, c2 = img2.shape
        assert c1 == c2 == 3

        h = max(h1, h2)
        w = w1 + w2
        img = np.zeros(shape=(h, w, 3), dtype=np.uint8)
        img[0: h1, 0: w1, :] = img1
        img[0: h2, w1: w1 + w2, :] = img2
        pt2[:, 0] += w1  # img2的kp坐标右移, 即x += w+int(w/3), y不变

        img_spindle = img.copy()  # initialized
        if draw_samples is None:
            if n > 100:
                index = list(range(n))
                random.shuffle(index)
                for i in range(n // 100):
                    img_i = img.copy()
                    if i * 100 + 100 >= n:
                        ind = index[i * 100: -1]
                    else:
                        ind = index[i * 100: i * 100 + 100]
                    for p1, p2 in zip(pt1[ind], pt2[ind]):
                        r = random.randint(0, 255)
                        g = random.randint(0, 255)
                        b = random.randint(0, 255)
                        x, y = p1
                        u, v = p2
                        img_i = cv2.line(img_i, (int(round(x)), int(round(y))),
                                         (int(round(u)), int(round(v))),
                                         color=(b, g, r), thickness=lwd)
                    if i == 0:
                        img_spindle = img_i
                    else:
                        img_spindle = np.r_[img_spindle, img_i]

            else:
                for p1, p2 in zip(pt1, pt2):
                    r = random.randint(0, 255)
                    g = random.randint(0, 255)
                    b = random.randint(0, 255)
                    x, y = p1
                    u, v = p2
                    img_spindle = cv2.line(img_spindle, (int(round(x)), int(round(y))),
                                           (int(round(u)), int(round(v))),
                                           color=(b, g, r), thickness=lwd)
        else:
            index = random.sample(list(range(n)), draw_samples)
            for p1, p2 in zip(pt1[index], pt2[index]):
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                x, y = p1
                u, v = p2
                img_spindle = cv2.line(img_spindle, (int(round(x)), int(round(y))),
                                       (int(round(u)), int(round(v))),
                                       color=(b, g, r), thickness=lwd)
        return img_spindle
