+++
title = "Types of Segmentation"
slug = "Types of Segmentation"
date = "2021-10-15"
+++

This project aims to compare 3 different types of segmentation: Boundary-based, Region-based and clustering. I've used tkinter to create a basic UI that lets the user decide on which segmentation to view.

![](/tos/img1.JPG)

**Boundary-based:**

In boundary based segmentation or edge-based segmentation, an edge filter is applied to the image, pixels are classified as edge or non-edge depending on the filter output, and pixels which are not separated by an edge are allocated to the same category.

![](/tos/img2.JPG)


**Region-based:**

In region-based segmentation algorithms operate iteratively by grouping together pixels which are neighbours and have similar values and splitting groups of pixels which are dissimilar in value. 

![](/tos/img3.JPG)


**Clustering:**

In K-Means clustering algorithm is an unsupervised algorithm and it is used to segment the interest area from the background. It clusters, or partitions the given data into K-clusters or parts based on the K-centroids.

![](/tos/img4.JPG)


## Code:

```python 
import pixellib
import cv2
import numpy as np
from pixellib.instance import instance_segmentation
from tkinter import *
win = Tk()
def exit():
  cv2.destroyAllWindows()
def button1():
  segment_image = instance_segmentation()
  segment_image.load_model("weights/mask_rcnn_coco.h5")
  segment_image.segmentImage("images/hill.jpg", output_image_name="segmentedimage.jpg", show_bboxes=True)

def button2():
  segmentimage = cv2.imread("segmentedimage.jpg")
  cv2.imshow('maskrcnn',segmentimage)

def button5():
  class Point(object):
    def __init__(self, x, y):
      self.x = x
      self.y = y

    def getX(self):
      return self.x

    def getY(self):
      return self.y

  def getGrayDiff(img, currentPoint, tmpPoint):
    return abs(int(img[currentPoint.x, currentPoint.y]) - int(img[tmpPoint.x, tmpPoint.y]))

  def selectConnects(p):
    if p != 0:
      connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), \
                  Point(0, 1), Point(-1, 1), Point(-1, 0)]
    else:
      connects = [Point(0, -1), Point(1, 0), Point(0, 1), Point(-1, 0)]
    return connects

  def regionGrow(img, seeds, thresh, p=1):
    height, weight = img.shape
    seedMark = np.zeros(img.shape)
    seedList = []
    for seed in seeds:
      seedList.append(seed)
    label = 1
    connects = selectConnects(p)
    while (len(seedList) > 0):
      currentPoint = seedList.pop(0)

      seedMark[currentPoint.x, currentPoint.y] = label
      for i in range(8):
        tmpX = currentPoint.x + connects[i].x
        tmpY = currentPoint.y + connects[i].y
        if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
          continue
        grayDiff = getGrayDiff(img, currentPoint, Point(tmpX, tmpY))
        if grayDiff < thresh and seedMark[tmpX, tmpY] == 0:
          seedMark[tmpX, tmpY] = label
          seedList.append(Point(tmpX, tmpY))
    return seedMark

  img = cv2.imread('images/hill.jpg', 0)
  seeds = [Point(10, 10), Point(82, 150), Point(20, 300)]
  binaryImg = regionGrow(img, seeds, 10)
  cv2.imshow('region growing', binaryImg)

def button4():
  import cv2
  import numpy as np
  import matplotlib.pyplot as plt
  image = cv2.imread("images/hill.jpg")
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  pixel_values = image.reshape((-1, 3))
  pixel_values = np.float32(pixel_values)
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
  ## number of clusters (K)
  k = 3
  _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
  centers = np.uint8(centers)

  labels = labels.flatten()
  segmented_image = centers[labels.flatten()]
  segmented_image = segmented_image.reshape(image.shape)
  cv2.imshow('kmeans', segmented_image)
#Making The Button
button1 =  Button(win, text="run", command=button1)
button3 = Button(win,text='view boundary based segmentation',command=button2)
button4 = Button(win,text='view clustering segmentation',command=button4)
button5 = Button(win,text='view region-based segmentation',command=button5)
button2 = Button(win,text="exit",command=exit)
#put on screen
button1.pack()
button3.pack()
button4.pack()
button5.pack()
button2.pack()
win.mainloop()

```
