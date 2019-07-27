import os
import sys
import numpy as np
import cv2

sys.path.append('./src')
from key_descriptor import SIFT
from ransac_matching import RansacMatching
from image_stitching import image_stitching

if __name__ == "__main__":
    filepath1 = "./image/Nikki_garden_01.jpg"
    filepath2 = "./image/Nikki_garden_03.jpg"
    img1 = cv2.imread(filepath1)
    img2 = cv2.imread(filepath2)
    print(img1.shape, img2.shape)

    # 1.
    sift = SIFT()
    kp1, des1 = sift.detect_and_compute(img1, False)
    kp2, des2 = sift.detect_and_compute(img2, False)
    p1, p2 = sift.get_matching_points(kp1, des1, kp2, des2, 0.995)
    M = p1.shape[0]
    print(M)

    # 2.
    img_spindle_name = "./result/Nikki_garden_match.jpg"
    if not os.path.exists(img_spindle_name):
        img_spindle = sift.draw_spindle(img1, img2, p1, p2, lwd=2)
        cv2.imwrite(img_spindle_name, img_spindle)

    img_spindle_sample_name = "./result/Nikki_garden_match_sample50.jpg"
    if not os.path.exists(img_spindle_sample_name):
        img_spindle_sample = sift.draw_spindle(img1, img2, p1, p2, lwd=2, draw_samples=50)
        cv2.imwrite(img_spindle_sample_name, img_spindle_sample)

    # 3.
    H_12_name = filepath2.split('/')[-1].split('.')[0]+'_H_12.npy'
    if os.path.exists(H_12_name):
        H_12 = np.load(H_12_name)
    else:
        ransac = RansacMatching()
        H_12 = ransac.get_homography(p1, p2, M // 5)  # H * image2 -> image1(without rotated)
        np.save(H_12_name, H_12)
    print(H_12)

    # H_21_name = filepath2.split('/')[-1].split('.')[0]+'_H_21.npy'
    # if os.path.exists(H_21_name):
    #     H_21 = np.load(H_21_name)
    # else:
    #     ransac = RansacMatching()
    #     H_21 = ransac.get_homography(p2, p1, M // 5)  # H * image1 -> image1(without rotated)
    #     np.save(H_21_name, H_21)
    # print(H_21)

    """"""
    # 4.
    img_stitched_name = "./result/Nikki_garden_stitched_filled.jpg"
    if not os.path.exists(img_stitched_name):
        stitch = image_stitching()
        img_stitched = stitch.get_img_stitched(img1, img2, p1, p2, H_12)
        cv2.imshow('image stitching', img_stitched)
        key = cv2.waitKey()
        if key == 27:
            cv2.destroyAllWindows()
        cv2.imwrite(img_stitched_name, img_stitched)
