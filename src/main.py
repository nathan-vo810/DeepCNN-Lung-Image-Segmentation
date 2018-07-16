import cv2
from src.DeepCNN_model import DeepCNN
from src.mobile_net_model import MobileNet
from src.vgg_model import VGGModel

import argparse
import os

DIR = os.path.dirname(__file__)
TEST_IMAGES = os.path.join(DIR, '../data/test/')


def get_args():
    # Create parser object
    parser = argparse.ArgumentParser(description="CLI for Lung Image Segmentation")

    parser.add_argument('-model', type=str, nargs=1,
                        metavar="model", default='DeepCNN', choices={'DeepCNN', 'MobileNet', 'VGG'},
                        help='Choose model type - DeepCNN/MobileNet/VGG')

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    if args.model == 'VGG':
        model = VGGModel()
    elif args.model == 'MobileNet':
        model = MobileNet()
    else:
        model = DeepCNN()

    model.build_model()
    model.predict(TEST_IMAGES + '21.jpg')


if __name__ == '__main__':
    main()
