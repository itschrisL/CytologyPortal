import sys
import os
import cv2 as cv
from cv2 import adaptiveThreshold
import numpy as np
import operator
import random
from PIL import Image, ImageEnhance

from matplotlib import pyplot as plt

import generator_utils as gen_utils

# TODO QuPath


# Concatenate many cells into one image
def concat_many_cell_images(img_list, file_name, output_dir,
                            shift=(0, 0), shift_mode='r'):
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

        # Create a random shift to the images to make them more realistic and spread out
        x_shift = random.randint(-30, 30)
        y_shift = random.randint(-30, 30)

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
        result = converter.enhance(1.75)

    # Save Images
    result.save(os.path.join(output_dir, "cell_clusters/" + file_name + ".png"))

    # Get image names for GTs
    img1_name, img2_name, img3_name = file_name.split("_")
    img_names = file_name.split("_")

    img1_seg = gen_utils.segmentation(img_list[0], (n_h, n_w))
    cv.imwrite(os.path.join(output_dir, "cell_clusters_gt/" + file_name + "_" + img1_name + "_gt.png"), img1_seg)
    img2_seg = gen_utils.segmentation(img_list[1], (n_h, n_w))
    cv.imwrite(os.path.join(output_dir, "cell_clusters_gt/" + file_name + "_" + img2_name + "_gt.png"), img2_seg)
    img3_seg = gen_utils.segmentation(img_list[2], (n_h, n_w))
    cv.imwrite(os.path.join(output_dir, "cell_clusters_gt/" + file_name + "_" + img3_name + "_gt.png"), img3_seg)

    for i in range(0, len(img_list)):
        img_seg = gen_utils.segmentation(img_list[i], (n_h, n_w))





# This will take a list of images and concatenate then together.
def concat_images_with_background(img_list, file_name):
    background = "C:/Users/lytle/OneDrive/Documents/UP/Cenek_Research/cell_photos/background_B.tif"
    background_img = cv.imread(background)

    # Create array to return all results, this includes overlapping cell image and their GTs
    # The overlapping cell will be the first then the
    img1 = img_list[0]
    img2 = img_list[1]

    # Get bgr values
    b, g, r = cv.split(img1)  # get b,g,r
    img1 = cv.merge([r, g, b])  # switch it to rgb

    # Get bgr values
    b, g, r = cv.split(img2)  # get b,g,r
    img2 = cv.merge([r, g, b])  # switch it to rgb

    # Convert to arrays
    img1 = Image.fromarray(img1)
    img2 = Image.fromarray(img2)

    # compute the size of the panorama
    nw, nh = map(max, map(operator.add, img2.size), img1.size)

    # Convert background image to image
    background_img = Image.fromarray(background_img[:nh, :nw, :])

    # paste img1 on top of img2
    newimg1 = Image.new('RGBA', size=(nw, nh), color=(0, 0, 0, 0))
    newimg1.paste(background_img)
    newimg1.paste(img1)

    # paste img2 on top of img1
    newimg2 = Image.new('RGBA', size=(nw, nh), color=(0, 0, 0, 0))
    newimg2.paste(background_img)
    newimg2.paste(img2)

    # blend with alpha=0.5
    result = Image.blend(newimg1, newimg2, alpha=0.5)

    converter = ImageEnhance.Contrast(result)
    result = converter.enhance(1.5)
    # plt.imshow(result_2), plt.show()

    result.save(os.path.join(OUTPUT_DIR, "cell_clusters/" + file_name + ".png"))

    # Get image names for GTs
    img1_name, img2_name = file_name.split("_")

    img1_seg = gen_utils.segmentation(img_list[0], (nh, nw))
    cv.imwrite(os.path.join(OUTPUT_DIR, "cell_clusters_gt/" + file_name + "_" + img1_name + "_gt.png"), img1_seg)
    img2_seg = gen_utils.segmentation(img_list[1], (nh, nw))
    cv.imwrite(os.path.join(OUTPUT_DIR, "cell_clusters_gt/" + file_name + "_" + img2_name + "_gt.png"), img2_seg)


# This will take a list of images and concatenate then together.
# Note: This method is being used to test new ways to concatenate images
def concat_images_with_background_v2(img_list, file_name):
    background = "C:/Users/lytle/OneDrive/Documents/UP/Cenek_Research/cell_photos/background_B.tif"
    background_img = cv.imread(background)

    # Create array to return all results, this includes overlapping cell image and their GTs
    # The overlapping cell will be the first then the
    img1 = img_list[0]
    img2 = img_list[1]

    # Get bgr values
    b, g, r = cv.split(img1)  # get b,g,r
    img1 = cv.merge([r, g, b])  # switch it to rgb

    # Get bgr values
    b, g, r = cv.split(img2)  # get b,g,r
    img2 = cv.merge([r, g, b])  # switch it to rgb

    seg_img1 = gen_utils.segmentation_cut(img_list[0])
    seg_img1 = Image.fromarray(seg_img1)

    seg_img2 = gen_utils.segmentation_cut(img_list[1])
    seg_img2 = Image.fromarray(seg_img2)

    # Convert to arrays
    img1 = Image.fromarray(img1)
    img2 = Image.fromarray(img2)

    # suppose img2 is to be shifted by `shift` amount
    shift = (0, 0)

    # compute the size of the panorama
    nw, nh = map(max, map(operator.add, img2.size, shift), img1.size)

    background_img = Image.fromarray(background_img[:nh, :nw, :])
    background_img = background_img.convert('RGBA')

    # paste img1 on top of img2
    newimg1 = Image.new('RGBA', size=(nw, nh), color=(0, 0, 0, 0))
    # newimg1.paste(background_img, shift)
    # newimg1.paste(img2, shift)
    newimg1.paste(seg_img1, (0, 0))

    print(f"w:{newimg1.width};h:{newimg1.height};mode:{newimg1.mode}")
    print(f"w:{background_img.width};h:{background_img.height};mode:{background_img.mode}")

    test = Image.blend(newimg1, background_img, alpha=0.5)
    plt.imshow(test), plt.show()

    # paste img2 on top of img1
    newimg2 = Image.new('RGBA', size=(nw, nh), color=(0, 0, 0, 0))
    # newimg2.paste(background_img, shift)
    # newimg2.paste(img1, (0, 0))
    # newimg2.paste(seg_img2, shift)
    newimg2.paste(img2)

    testimg = Image.new('RGBA', size=(nw, nh), color=(0, 0, 0, 0))
    ar = np.array(testimg)
    img1_seg = np.array(seg_img1)
    back = np.array(background_img)

    # for x in range(0, len(ar)):
    #     for y in range(0, len(ar[0])):
    #         if img1_seg[x][y][3] == 255:
    #             ar[x][y] = img1_seg[x][y]
    #         else:
    #             ar[x][y] = back[x][y]
    #
    # plt.imshow(ar), plt.show()

    # print(f"w:{results.width};h:{results.height}")
    # print(f"w:{background_img.width};h:{background_img.height}")


# Method that will concatenate two cell images into one
# It first cuts out the image then overlaps the two images together
# Then the images are pasted onto a background image
def concat_images_by_segmentation(img1, img2):
    # Get background image
    background = "C:/Users/lytle/OneDrive/Documents/UP/Cenek_Research/cell_photos/background_B.tif"
    background_img = cv.imread(background)

    # Get bgr values
    b, g, r = cv.split(img1)  # get b,g,r
    img1 = cv.merge([r, g, b])  # switch it to rgb

    img1_seg = gen_utils.segmentation_cut(img1)
    plt.imshow(img1_seg), plt.show()

    # Get bgr values
    b, g, r = cv.split(img2)  # get b,g,r
    img2 = cv.merge([r, g, b])  # switch it to rgb

    img2_seg = gen_utils.segmentation_cut(img2)

    # compute the size of the panorama
    nw, nh = max(img1.shape[0], img2.shape[0]), max(img1.shape[1], img2.shape[1])

    # Convert background image to image and resize it to fit our height and width
    background_img = Image.fromarray(background_img[:nh, :nw, :])
    background_img = background_img.convert("RGBA")

    # create new images
    newimg1 = Image.new("RGBA", size=(nh, nw), color=(0, 0, 0, 0))
    # newimg1 = np.zeros(shape=(nw, nh, 4))
    newimg1 = np.array(newimg1)
    back = np.array(background_img)
    alpha = 0.5
    for x in range(0, len(newimg1)):
        for y in range(0, len(newimg1[0])):
            if img1_seg[x][y][3] == 255 and img2_seg[x][y][3] == 255:
                # Blend images together
                # Based on this formula: out = image1 * (1.0 - alpha) + image2 * alpha
                pixels_1 = img1_seg[x][y]
                pixels_2 = img2_seg[x][y]
                out = pixels_1 * (1.0 - alpha) + pixels_2 * alpha
                newimg1[x][y] = out
            elif img1_seg[x][y][3] == 255:
                newimg1[x][y] = img1_seg[x][y]
            elif img2_seg[x][y][3] == 255:
                newimg1[x][y] = img2_seg[x][y]
            else:
                newimg1[x][y] = back[x][y]

    newimg1 = Image.fromarray(newimg1, mode="RGBA")
    plt.imshow(newimg1), plt.show()


# argv[1] = input Dir
# argv[2] = output Dir
if __name__ == "__main__":
    # Check if there are input arguments
    if len(sys.argv) >= 3:
        # Save arguments
        INPUT_DIR = sys.argv[1]
        OUTPUT_DIR = sys.argv[2]

        if not os.listdir(INPUT_DIR):
            print("Input path is not a valid directory")
            print("Got: " + INPUT_DIR)

        elif not os.listdir(OUTPUT_DIR):
            print("Output path is not a valid directory")
            print("Got: " + OUTPUT_DIR)

        else:
            # List of images and GTs [(original image, GT)]
            img_list = []

            file_count = 0
            print("Getting images from : " + INPUT_DIR)
            # Get all images in input directory
            for file in os.listdir(INPUT_DIR):
                if file.endswith(".tif") or file.endswith(".png") or file.endswith(".jpg"):
                    # Read image from file and get gt
                    image = cv.imread(os.path.join(INPUT_DIR, file))
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


            # for n in range(0, len(img_list)):
            #     break
            #     img1 = img_list[n][1]
            #     img1_name = img_list[n][0]
            #     for i in range(n+1, len(img_list)):
            #         img2 = img_list[i][1]
            #         img2_name = img_list[i][0]
            #         name = img1_name + "_" + img2_name
            #         concat_images_with_background([img1, img2], name)

            num = 20
            np.random.shuffle(img_list)
            for n in range(0, num):
                print(f"{(n/num)*100}%")
                img1 = img_list[n][1]
                img1_name = img_list[n][0]
                for i in range(n + 1, num):
                    img2 = img_list[i][1]
                    img2_name = img_list[i][0]
                    for j in range(i + 1, num):
                        img3 = img_list[j][1]
                        img3_name = img_list[j][0]

                        img_name = img1_name + "_" + img2_name + "_" + img3_name

                        concat_many_cell_images([img1, img2, img3], img_name, OUTPUT_DIR)

            n = 20  # Number of images to generate
            c = 3  # Number of overlapping cells per image
            np.random.shuffle(img_list)  # Shuffle list
            for i in range(0, n):
                cells = img_list[i:i+c][1]
                call_names = img_list[i:i+c][0]



            # test = concat_many_cell_images([img1, img2, img3], "")


    else:
        print("Invalid Arguments Length")
        print("argv[1]: Input Directory")
        print("argv[2]: Output Directory")
        print("argv[3]: Show Image (optional)")



