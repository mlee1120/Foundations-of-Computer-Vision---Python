"""
file: yolo.py
description: CSCI 631 Final Project
language: python3
author: Owen Shriver ofs9424@rit.edu
"""

import torch
import csv
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from coco_classes import coco_classes

def iou(b1, b2):
    # x1, y1, x2, y2
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    u = (b1[2] - b1[0]) * (b1[3] - b1[1]) + (b2[2] - b2[0]) * (b2[3] - b2[1]) - i
    return i / u

# Images
dir = "images/"
infofile = "information.csv"

imgs = os.listdir(dir)

gt = {}
with open(infofile) as information:
    r = csv.reader(information)
    for row in r:
        fname = row[0] + ".jpg"
        if fname in imgs and row[9] in coco_classes.keys():
            frames = row[12]
            f_no = int(row[2])
            if fname not in gt.keys():
                gt[fname] = [f_no]
            fr_nos = [int(s[-6:]) for s in frames.split("': '")[:-1]]
            if f_no in fr_nos:
                i = fr_nos.index(f_no)
                loc = [int(x) for x in frames.split('[')[i + 1].split(']')[0].split(", ")]
                gt[fname].append([row[9]] + loc)

batch_size = 32
thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

i = 0
total_objects = 0
found = [0 for x in thresholds]
average_iou = [0.0 for x in thresholds]
accuracy = [0 for x in thresholds]
conf_in_correct = [0 for x in thresholds]
conf_in_incorrect = [0 for x in thresholds]

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

while i < (len(imgs)/ batch_size) * batch_size:
    # Inference
    results = model([dir + p for p in imgs[i:i+batch_size]])

    for j in range(len(results.xyxy)):
        for obj in gt[imgs[i+j]][1:]:
            total_objects += 1
            highest_pred = 0
            highest_iou = 0
            for k in range(len(np.array(results.xyxy[j]))):
                prediction = results.xyxy[j][k]
                computed_iou = iou(obj[1:], prediction[:4])
                if computed_iou > highest_iou:
                    highest_pred = k
                    highest_iou = computed_iou
            for tx in range(len(thresholds)):
                threshold = thresholds[tx]
                if highest_iou > threshold:
                    found[tx] += 1
                    average_iou[tx] += highest_iou
                    if results.xyxy[j][highest_pred][5] == coco_classes[obj[0]]:
                        accuracy[tx] += 1
                        conf_in_correct[tx] += results.xyxy[j][highest_pred][4]
                    else:
                        conf_in_incorrect[tx] += results.xyxy[j][highest_pred][4]

    i += batch_size

plt.plot(thresholds, np.divide(found, total_objects))
plt.xlabel("Threshold")
plt.ylabel("Proportion of objects found")
plt.savefig("found.png")
plt.clf()

plt.plot(thresholds, np.divide(average_iou, found))
plt.xlabel("Threshold")
plt.ylabel("Average IOU of detected objects")
plt.savefig("average_iou.png")
plt.clf()

plt.plot(thresholds, np.divide(accuracy, found))
plt.xlabel("Threshold")
plt.ylabel("Accuracy")
plt.savefig("accuracy.png")
plt.clf()

plt.plot(thresholds, np.divide(conf_in_correct, found))
plt.xlabel("Threshold")
plt.ylabel("Average confidence in correct predictions")
plt.savefig("conf_correct.png")
plt.clf()

plt.plot(thresholds, np.divide(conf_in_incorrect, np.subtract(found, accuracy)))
plt.xlabel("Threshold")
plt.ylabel("Average confidence in incorrect predictions")
plt.savefig("conf_incorrect.png")
plt.clf()

plt.plot(thresholds, np.divide(accuracy, total_objects))
plt.xlabel("Threshold")
plt.ylabel("Proportion of objects accurately classified")
plt.savefig("detect_and_classify.png")
plt.clf()
