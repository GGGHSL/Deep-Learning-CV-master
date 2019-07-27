import numpy as np
import cv2


class image_stitching:
    def __get_img_transformed(self, dst, H):
        h, w, _ = dst.shape
        corners = np.array([
            [0, 0, 1],  # tl
            [h - 1, 0, 1],  # dl
            [0, w - 1, 1],  # tr
            [h - 1, w - 1, 1]  # dr
        ]).T
        trans_corners = np.matmul(H, corners)
        trans_corners = np.divide(trans_corners, trans_corners[-1, :])[0:2]
        y_lim = round(max(trans_corners[0]) - min(trans_corners[0]))
        x_lim = round(max(trans_corners[1]) - min(trans_corners[1]))
        print("lims", int(x_lim), int(y_lim))
        H_invs = np.linalg.inv(H)
        src = cv2.warpPerspective(dst, M=H_invs,
                                  # dsize=(dst.shape[1], dst.shape[0]),
                                  dsize=(int(x_lim), int(y_lim)),)
        cv2.imshow(' ', src)
        key = cv2.waitKey()
        if key == 27:
            cv2.destroyAllWindows()
        return src

    def __get_point_transformed(self, p, H):
        H_invs = np.linalg.inv(H)
        u, v = p
        dst = np.array([u, v, 1]).reshape(3, 1)
        x, y, w = np.matmul(H_invs, dst)
        src = np.array([x / w, x / w])
        return src

    def get_img_stitched(self, img1, img2, pt1, pt2, H_12):  # , H_21
        # img1 = self.get_img_transformed(img1, H_21)
        # print(img1.shape)
        h1, w1, _ = img1.shape
        img2 = self.__get_img_transformed(img2, H_12)
        print("img2", img2.shape)
        h2, w2, _ = img2.shape
        n = pt1.shape[0]
        delta_u, delta_v = 0, 0
        for p1, p2 in zip(pt1, pt2):
            # p1 = self.__get_point_ptransformed(p1, H_21)
            p2 = self.__get_point_transformed(p2, H_12)
            x, y = p1
            u, v = p2
            delta_u += (u.item() - x.item())
            delta_v += (v.item() - y.item())
        delta_u /= n  # -367
        delta_u += w1  # += 1288
        delta_v /= n

        delta_u = round(delta_u)
        delta_v = round(delta_v)
        print("delta", delta_u, delta_v)

        h = max(h1, h2 - delta_v)  #
        w = w1 + w2 - delta_u
        img = np.zeros(shape=(h, w, 3), dtype=np.uint8)
        print("img", img.shape)
        img[0: h1, 0: w1, :] = img1
        print("patch", )
        img[abs(delta_v) - 2: -2, w1 - delta_u:, :] = img2

        # Fill block
        for i in range(h1):
            for j in range(w1):
                b, g, r = img[i, j, :]
                if b == 0 and g == 0 and r == 0:
                    img[i, j, :] = img1[i, j, :]
        img = img[0: max(h1, h2)]
        return img
