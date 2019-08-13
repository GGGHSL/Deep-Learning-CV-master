import os
from PIL import Image
import numpy as np
import copy


def get_expanded_info(root, info):
    img_name = info[0]
    img_path = root + "\\data\\" + "I\\" + img_name
    if not os.path.exists(img_path):
        img_path = root + "\\data\\" + "II\\" + img_name
    if not os.path.exists(img_path):
        return None

    img = Image.open(img_path)
    w, h = img.size

    keypoints = info[5:47].astype(np.float32).reshape(21, 2)
    x1, y1, x2, y2 = info[1:5].astype(np.float32)  # 左上角, 右下角
    if np.any(keypoints < 0) or np.any(keypoints[:, 0] >= h) or np.any(keypoints[:, 1] >= w):
        return None
    b_w, b_h = x2 - x1, y2 - y1
    e_x1, e_y1, e_x2, e_y2 = x1 - 0.25 * b_w, y1 - 0.25 * b_h, x2 + 0.25 * b_w, y2 + 0.25 * b_h
    if not (0 <= e_x1 < e_x2 < w and 0 <= e_y1 < e_y2 < h):
        return None

    info[1:5] = np.array([e_x1, e_y1, e_x2, e_y2]).astype(np.str)
    keypoints -= np.array([e_x1, e_y1])
    info[5:47] = keypoints.ravel().astype(np.str)
    return info


def generate_train_test(root, test_ratio=0.3):
    path = [root + "\\data\\I", root + "\\data\\II"]
    src = np.r_[np.loadtxt(path[0] + "\\" + "label.txt", dtype=np.str),
                np.loadtxt(path[1] + "\\" + "label.txt", dtype=np.str)]
    n = src.shape[0]
    src = src[np.random.shuffle(list(range(n)))][0]
    print("Total data size: ", n)

    del_idx = []
    train_size = int(n * (1 - test_ratio))
    print("Training size: ", train_size)
    if not os.path.exists("train.txt"):
        train_file = open("train.txt", 'w')
        for idx in range(train_size):
            tmp = copy.deepcopy(src[idx])
            info = get_expanded_info(root, tmp)
            if info is None:
                del_idx.append(idx)
                continue
            str_info = ''
            for i in info:
                str_info += i + " "
            train_file.write(str_info[:-1] + '\n')
        train_file.close()

    print("Test size: ", n - train_size)
    if not os.path.exists("test.txt"):
        test_file = open("test.txt", 'w')
        for idx in range(train_size, n):
            tmp = copy.deepcopy(src[idx])
            info = get_expanded_info(root, tmp)
            if info is None:
                del_idx.append(idx)
                continue
            str_info = ''
            for i in info:
                str_info += i + " "
            test_file.write(str_info[:-1] + '\n')
        test_file.close()

    remain_idx = [i for i in list(range(n)) if i not in del_idx]
    remain = src[remain_idx]
    if not os.path.exists("all.txt"):
        file = open("all.txt", 'w')
        for line in remain:
            str_line = ''
            for i in line:
                str_line += i + " "
            file.write(str_line[:-1] + '\n')
        file.close()


if __name__ == "__main__":
    root = os.getcwd()
    print(root)
    generate_train_test(root)
