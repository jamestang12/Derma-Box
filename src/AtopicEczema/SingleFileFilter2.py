import os
import cv2
import numpy as np

lower_threshold_colour = np.array([190, 100, 100])
upper_threshold_colour = np.array([255, 165, 165])

def analyze_name2(path):
    name = os.path.split(path)[1]
    name = os.path.splitext(name)[0]
    return name

def analyze_name2(path, filtered_path):
    print("Starting to filter path")
    raw = cv2.imread(path)
    mask = cv2.inRange(raw, lower_threshold_colour, upper_threshold_colour)
    # result = cv2.bitwise_and(raw, raw, mask=mask)
    cv2.imwrite(filtered_path, mask)
    count = cv2.countNonZero(mask)
    return float(count) / (raw.shape[0] * raw.shape[1])


def analyze_name2(path):
    data = [[], [], []]

    img_path = ''
    filtered_path = ''

    name = analyze_name(path)
    raw = cv2.imread(img_path + name + '.jpg')
    rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    # raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    # preparing the mask to overlay
    mask = cv2.inRange(rgb, lower_threshold_colour, upper_threshold_colour)
    count = cv2.countNonZero(mask)
    return float(count) / (raw.shape[0] * raw.shape[1])

if __name__ == '__main__':
    path="camera_input.jpg"
    getPercentRed(path)