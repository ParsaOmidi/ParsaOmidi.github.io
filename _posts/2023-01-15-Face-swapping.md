---
layout: post
title: Face swapping between two images
date: 2023-01-15 15:09:00
description: "Using statistical methods to perform sentiment analysis on a dataset of reviews might be simpler than you think."
tags: MachineVision OpenCV Python

giscus_comments: true
---
This is a Python script that makes use of the OpenCV and dlib libraries to perform face swapping between two images. The script reads in two images, finds the facial landmarks on each image using dlib or OpenCV's Haar cascades, aligns the faces using Procrustes analysis, and finally swaps the faces by blending the two images together.


{% highlight Python linenos %}

# Import necessary libraries
import cv2
import dlib
import numpy
from time import sleep
import sys

# Pretrained model that predicts facial feature rectangles
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
SCALE_FACTOR = 1
FEATHER_AMOUNT = 11

# Define different facial feature points for later use
FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

# Points used to align images
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                               RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

# Points from the second image to overlay on the first
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]

# Amount of blur to use during colour correction, as a fraction of the
# pupillary distance.
COLOUR_CORRECT_BLUR_FRAC = 0.6

# Load the face cascade classifier and the dlib frontal face detector
cascade_path='Haarcascades/haarcascade_frontalface_default.xml'
cascade = cv2.CascadeClassifier(cascade_path)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)


def get_landmarks(im, dlibOn):
    """
    Get facial landmarks from the image using either dlib or OpenCV.

    :param im: Input image to get landmarks from.
    :param dlibOn: Boolean indicating whether to use dlib or OpenCV for facial detection.
    :return: Matrix containing the x and y coordinates of the facial landmarks.
    """
    # If using dlib, detect facial landmarks using the dlib frontal face detector
    if (dlibOn == True):
        rects = detector(im, 1)
        if len(rects) > 1:
            return "error"
        if len(rects) == 0:
            return "error"
        return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

    # Otherwise, use OpenCV to detect facial landmarks
    else:
        rects = cascade.detectMultiScale(im, 1.3,5)
        if len(rects) > 1:
            return "error"
        if len(rects) == 0:
            return "error"
        x,y,w,h =rects[0]
        rect=dlib.rectangle(x,y,x+w,y+h)
        return numpy.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])


def annotate_landmarks(im, landmarks):
    """
    Annotate facial landmarks on the image.

    :param im: Input image to annotate landmarks on.
    :param landmarks: Matrix containing the x and y coordinates of the facial landmarks.
    :return: Copy of the input image with facial landmarks annotated.
    """
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT


{% endhighlight %}
