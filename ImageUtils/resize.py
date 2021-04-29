import argparse
import cv2
import numpy as np
from pathlib import Path
import sys

IMAGE_DIM=(224, 224)


def parse_args():
    parser = argparse.ArgumentParser(description="resize GE_Grabber images")
    parser.add_argument('--input', '-i', type=str, required=True, help='input directory containing images to resize')
    parser.add_argument('--output', '-o', type=str, required=True, help='output directory to contain resized images')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    in_dir = Path(args.input)
    out_dir = Path(args.output)

    if not in_dir.is_dir():
        sys.exit(f'Directory "{in_dir}" does not exist')

    if not out_dir.is_dir():
        out_dir.mkdir()

    print(f'Resizing images from "{in_dir}" to "{out_dir}"')

    for filename in in_dir.iterdir():
        img = cv2.imread(str(filename))
        img_resized = cv2.resize(img, IMAGE_DIM)
        out_filename = out_dir / filename.name
        cv2.imwrite(str(out_filename), img_resized)
        # print(f'{filename}: {img.shape} to {out_filename}: {img_resized.shape}')

# 224
