"""
file: main.py
description: CSCI 631 Final Project
language: python3
author: Michael Lee ml3406@RIT.EDU
"""

import os
import torch
from PIL import Image

# used to read in data (images)
images = []
# select the video (for multiple-frame detection)
a = "Sport_7"


def load_model():
    """
    This function loads model
    :return: the model
    """

    # Model (choose one below)
    m = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5m')
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5l')
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5x')
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5s6')
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5m6')
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5l6')
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5x6')
    return m


def read_images():
    """
    This function reads in the data (images)
    """

    # read images (multiple frames)
    for img in os.listdir("data/frames/" + a):
        images.append(Image.open("data/frames/" + a + "/" + img))

    # single frame
    '''
    for img in os.listdir("data/frame"):
        images.append(Image.open("data/frame/" + img))
    '''


def test_and_save(m):
    """
    This function tests the data with the model and classifies them.
    :param m: the model
    """

    os.mkdir("Results/" + a)
    results = m(images, size=640)
    # to draw bounding boxes with confidence scores
    results.save("Results/" + a)

    # the code below can be use to save images to different folders according their class
    # with the highest confidence score (classify images)
    '''
    for i in range(0, len(results), 1):
        im = Image.fromarray(results.imgs[i])
        path = ""
        if len(results.xyxy[i]) == 0:
            if not os.path.isdir("Results/Classification/None"):
                os.mkdir("Results/Classification/None")
            path = "Results/Classification/None/"
        else:
            if not os.path.isdir("Results/Classification/" + str(int(results.xyxy[i][0][5]))):
                os.mkdir("Results/Classification/" + str(int(results.xyxy[i][0][5])))
            path = "Results/Classification/" + str(int(results.xyxy[i][0][5])) + "/"
        im.save(path + results.files[i])
    '''

    # print the bounding boxes information and the confidence scores
    for i in results.xyxy:
        print(i)
    # results.xyxy[1]  # img1 predictions (tensor)
    #      xmin    ymin    xmax   ymax  confidence  class    name
    # 0  749.50   43.50  1148.0  704.5    0.874023      0  person
    # 1  433.50  433.50   517.5  714.5    0.687988     27     tie
    # 2  114.75  195.75  1095.0  708.0    0.624512      0  person
    # 3  986.00  304.00  1028.0  420.0    0.286865     27     tie


"""
main conditional guard
The following condition checks whether we are running as a script.
If the file is being imported, don't run the test code.
"""
if __name__ == '__main__':
    model = load_model()
    read_images()
    test_and_save(model)
