import argparse
import sys
sys.path.append('./src/')
import warnings
warnings.filterwarnings("ignore")
from utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--opts', type=str, choices=["random", "show_all"])
    args = parser.parse_args()

    data_path = './data/'
    file_list = os.listdir(data_path)
    for file in file_list:
        file_path = os.path.join(data_path, file)
        if os.path.isfile(file_path):
            print('\n', file_path)
            data_aug = DataAugmentation(file_path)
            if args.opts == 'show_all':
                data_aug.show_all_augmentations(dpi=1000)
            else:
                data_aug.random_augmentation(dpi=1000)
