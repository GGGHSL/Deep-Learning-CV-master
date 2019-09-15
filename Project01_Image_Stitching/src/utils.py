import cv2
import random
import numpy as np


def draw_spindle(img1, img2, pt1, pt2):
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
    w = max(w1, w2)
    img = np.zeros(shape=(h, 2*w+int(w/3), 3), dtype=np.uint8)
    img[0: h1, 0: w1, :] = img1
    img[0: h2, w+int(w/3): w+int(w/3)+w2, :] = img2
    pt2[:, 0] += w + int(w/3)  # img2的kp坐标右移, 即x += w+int(w/3), y不变

    for p1, p2 in zip(pt1, pt2):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        img = cv2.line(img, tuple(p1), tuple(p2), color=(b, g, r))
    return img
