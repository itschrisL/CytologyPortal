import os
import argparse
import cv2 as cv
import numpy as np
from operator import itemgetter

import generator_utils as gen_utils


# Handle user input
parser = argparse.ArgumentParser(description="Pre-process image data for pipeline rust segmentation")
parser.add_argument('-i', '--input', type=str, default='./images/input_images',
                    help='A path to the directory containing the input images')
parser.add_argument('-o', '--output', type=str, default='./images/output_images',
                    help='A path to the directory the output images are going to be dumped')
parser.add_argument('-n', '--num', type=int, default=-1,
                    help='The number of images to create')
parser.add_argument('-c', '--combinations', type=int, default=3,
                    help='The number of images to combine together')
parser.add_argument('-s', '--shift', type=int, default=0,
                    help='The amount to shift over')
parser.add_argument('-st', '--shift_xy', type=tuple, default=(0, 0))
parser.add_argument('-cc', '--color_contrast', type=float, default=1.75)
args = parser.parse_args()


input_dir = args.input
output_dir = args.output
num_img = args.num
comb = args.combinations
shift = args.shift


if not os.listdir(input_dir):
    print("Input path is not a valid directory")
    print("Got: " + input_dir)

elif not os.listdir(output_dir):
    print("Output path is not a valid directory")
    print("Got: " + output_dir)

else:
    # List of images and GTs [(original image, GT)]
    img_list = []

    file_count = 0
    print("Getting images from : " + input_dir)
    # Get all images in input directory
    for file in os.listdir(input_dir):
        if file.endswith(".tif") or file.endswith(".png") or file.endswith(".jpg"):
            # Read image from file and get gt
            image = cv.imread(os.path.join(input_dir, file))
            gt = gen_utils.segmentation(image)

            # Flip image
            horizontal_img = cv.flip(image.copy(), 0)
            vertical_img = cv.flip(image.copy(), 1)
            both_img = cv.flip(image.copy(), -1)

            horizontal_img_gt = gen_utils.segmentation(horizontal_img)
            vertical_img_gt = gen_utils.segmentation(vertical_img)
            both_img_gt = gen_utils.segmentation(both_img)

            new_file_name = file.split(".")[0]

            # Add images to img list
            img_list.append((str(file_count), image, gt))
            img_list.append((str(file_count) + "h", horizontal_img, horizontal_img_gt))
            img_list.append((str(file_count) + "v", vertical_img, vertical_img_gt))
            img_list.append((str(file_count) + "b", both_img, both_img_gt))

            # Increase file Count
            file_count += 1

    print(f"There are {len(img_list)} images to blend from {file_count} images")

    if not os.path.exists(os.path.join(output_dir, "cell_clusters/")):
        os.mkdir(os.path.join(output_dir, "cell_clusters/"))

    if not os.path.exists(os.path.join(output_dir, "cell_clusters_gt/")):
        os.mkdir(os.path.join(output_dir, "cell_clusters_gt/"))

    if num_img == -1:
        num_img = len(img_list) - comb

    n = 20  # Number of images to generate
    c = 3  # Number of overlapping cells per image
    np.random.shuffle(img_list)  # Shuffle list
    for i in range(0, n):
        imgs = img_list[i: i + c]
        cells = [im[1] for im in imgs]
        cell_names = [n[0] for n in imgs]
        separator = "_"
        img_name = separator.join(np.array(cell_names))

        gen_utils.concat_many_cell_images(cells, img_name, output_dir, shift=(shift, shift))
