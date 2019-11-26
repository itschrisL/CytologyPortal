import sys
import os
import cv2 as cv
from cv2 import adaptiveThreshold
import numpy as np
from PIL import Image, ImageEnhance
import random

from matplotlib import pyplot as plt


# Another way to cut out cell images from the background
# The article that I found this is bellow
# https://docs.opencv.org/3.4/d8/d83/tutorial_py_grabcut.html
def grab_cut_cell(img):
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (50, 50, 450, 290)
    cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]
    plt.imshow(img), plt.colorbar(), plt.show()


def segmentation_cut(img, new_size=None):
    # Convert img to gray scale
    gray = cv.cvtColor(img.copy(), cv.COLOR_BGR2GRAY)
    # Preform segmentation based on the threshold function
    ret, seg_img = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    seg_img = cv.GaussianBlur(seg_img, (5, 5), 0)

    # Now convert original image into RGBA and only display the cell
    # This is done by setting the alpha channel to 0
    result = cv.cvtColor(img.copy(), cv.COLOR_RGB2RGBA)
    for x in range(0, img.shape[0]):
        for y in range(0, img.shape[1]):
            if seg_img[x][y] != 255:
                result[x][y] = [0, 0, 0, 0]
            else:
                result[x, y, 3] = 255

    if new_size is not None:
        bot_border = new_size[0] - result.shape[0]
        right_border = new_size[1] - result.shape[1]
        result = cv.copyMakeBorder(result, 0, bot_border, 0, right_border, cv.BORDER_CONSTANT, value=[0, 0, 0])
    return result


# Preform image segmentation on cell
# Parameters:
#   img - the image to segment
#   new_size - size of the image to be produced
#              Any new additions will be added as black backgrounds
#              Set to None by default meaning that it won't height or width to the image
def segmentation(img, new_size=None):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, seg_img = cv.threshold(gray, 100, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    # seg_img = cv.adaptiveThreshold(gray, 50, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 1001, 13)
    # seg_img = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 101, 21)
    seg_img = cv.GaussianBlur(seg_img, (5, 5), 0)
    # plt.imshow(seg_img), plt.show()
    if new_size is not None:
        bot_border = new_size[0] - seg_img.shape[0]
        right_border = new_size[1] - seg_img.shape[1]
        seg_img = cv.copyMakeBorder(seg_img, 0, bot_border, 0, right_border, cv.BORDER_CONSTANT, value=[0, 0, 0])
    return seg_img


# Concatenate many cells into one image
def concat_many_cell_images(img_list, file_name, output_dir,
                            alpha_enhance=1.75, shift=None):
    background = "background_B.tif"
    background_img = cv.imread(background)

    # variables for new height and width
    n_h, n_w = 0, 0
    # Convert the images from bgr to rgb
    # Also get the new height and width
    rgb_img_list = []
    for img in img_list:
        b, g, r = cv.split(img)
        new_img = cv.merge([r, g, b])
        new_img = Image.fromarray(new_img)

        # Add new image to rgb image array
        rgb_img_list.append(new_img)

        # Update new height and width
        if new_img.height >= n_h:
            n_h = new_img.height
        if new_img.width >= n_w:
            n_w = new_img.width

    # Crop background image with new height and width
    new_background_img = Image.fromarray(background_img[:n_h, :n_w, :])

    to_blend = []
    for img in rgb_img_list:
        new_img = Image.new('RGBA', size=(n_w, n_h), color=(0, 0, 0, 0))

        if shift is None:
            # Create a random shift to the images to make them more realistic and spread out
            x_shift = random.randint(-30, 30)
            y_shift = random.randint(-30, 30)
        else:
            # Shift be x and y amount
            x_shift = shift[0]
            y_shift = shift[1]

        # make sure the new shift doesn't go over
        if n_h <= img.height + x_shift <= 0:
            x_shift = 0

        if n_w <= img.width + y_shift <= 0:
            y_shift = 0

        new_img.paste(new_background_img)
        new_img.paste(img, (x_shift, y_shift))
        to_blend.append(new_img)

    # Make the first image the starting point
    result = to_blend[0]

    for i in range(1, len(to_blend)):
        result = Image.blend(result, to_blend[i], alpha=0.5)

        # Create an enhancer element
        converter = ImageEnhance.Contrast(result)
        result = converter.enhance(alpha_enhance)

    # Save Images
    result.save(os.path.join(output_dir, "cell_clusters/" + file_name + ".png"))

    # Get image names for GTs
    img_names = file_name.split("_")

    # Save ground truths
    for i in range(0, len(img_list)):
        img_seg = segmentation(img_list[i], (n_h, n_w))
        cv.imwrite(os.path.join(
            output_dir, 'cell_clusters_gt/' + file_name +
            '_' + img_names[i] + '_gt.png'),
            img_seg)
