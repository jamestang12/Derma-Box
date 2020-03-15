# -*- coding: utf-8 -*-
import os
import cv2
import glob
import numpy as np
import csv
import random

def analyze_name(path):
    name = os.path.split(path)[1]
    name = os.path.splitext(name)[0]
    return name

# 209, 139, 113
# lower_threshold_colour = np.array([20.0, 30.0, 50.0])
# upper_threshold_colour = np.array([80.0, 100.0, 255.0])
lower_threshold_colour = np.array([190, 100, 100])
upper_threshold_colour = np.array([255, 165, 165])

if __name__ == '__main__':
    data = [[], [], []]

    # img_path = './rawImages/'
    # cornea_path = './corneaLabels/'
    # ulcer_path = './ulcerLabels/'
    # cornea_overlay_path = './corneaOverlay/'
    # ulcer_overlay_path = './ulcerOverlay/'
    img_path = 'rawImages/'
    filtered_path = 'filteredImages/'

    # generate overlay images
    images = sorted(glob.glob(img_path + '*.jpg'))
    print("Starting to filter raw images!")
    i = 0

    for path in images:
        name = analyze_name(path)
        raw = cv2.imread(img_path + name + '.jpg')
        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        # raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        # preparing the mask to overlay
        mask = cv2.inRange(rgb, lower_threshold_colour, upper_threshold_colour)

        # The black region in the mask has the value of 0,
        # so when multiplied with original image removes all non-blue regions
        result = cv2.bitwise_and(raw, raw, mask=mask)
        # cv2.imwrite(filtered_path + name + '.jpg', result)
        cv2.imwrite(filtered_path + name + '.jpg', mask)
        # cv2.imshow('frame', frame)
        # cv2.imshow('mask', mask)
        # cv2.imshow('result', result)
        count = cv2.countNonZero(mask)
        data[0].append(i)
        data[1].append(float(count) / (raw.shape[0] * raw.shape[1]))

        # if (i <= 52):
        data[2].append(1)
        # else:
        #    data[2].append(0)

        i += 1
        print("Filtered " + path)

    for i in range(0, 500):
        data[0].append(101 + i)
        data[1].append(random.random() / 500.0)

        # if (i <= 52):
        data[2].append(0)

    print("Finished filtering.")
    print("Writing to filtered_image_data.csv.")

    with open('filtered_image_data.csv', mode='w') as employee_file:
        csv_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["Name", "Percent_Red", "Is_Sick"])

        for i in range(0, len(data[0])):
            csv_writer.writerow([data[0][i], data[1][i], data[2][i]])

    print("Finished writing to filtered_image_data.csv.")
    print("Done all!")