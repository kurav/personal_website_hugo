+++
title = "Colour picker"
slug = "Colour picker"
date = "2021-07-03"
+++

This project aims to determine the (r,g,b) value of a selected pixel. It also tries to create a threshold and only highlight the colour of the pixel that has been clicked on.

## The RGB colour model
A color in the RGB color model can be described by indicating how much of each of red, green, and blue is included. The color is expressed as (r,g,b), each component of which can vary from zero to a defined maximum value. If all the components are at zero the result is black; if all are at maximum, the result is white.

## Output

The output of various clicks on the given image:

![](/tos/img5.JPG)


The output of thresholding the selected colour:

![](/tos/img6.JPG)


## Code

```python 
import cv2
import numpy as np

img = cv2.imread("images/hand.jpg")

def colorpickwindow(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        colorsBGR = img[y, x]
        colorsRGB = tuple(reversed(colorsBGR))
        print("RGB Value at ({},{}):{} ".format(x, y, colorsRGB))
        if colorsRGB == (242, 80, 34):
            boundaries = [
                ([15, 60, 230], [50, 100, 250])
            ]
        elif colorsRGB == (127, 186, 0):
            boundaries = [
                ([0, 180, 110], [10, 190, 140])
                ]
        elif colorsRGB == (1, 164, 239):
            boundaries = [
                ([238, 163, 0], [240, 165, 2])
                ]
        elif colorsRGB == (255, 185, 1):
            boundaries = [
                ([0, 170, 254], [2, 200, 255])
                ]
        elif colorsRGB == (255, 255, 255):
            boundaries = [
                ([255, 255, 255], [255, 255, 255])
                ]
        else:
            boundaries = [
                ([105, 105, 105], [130, 130, 130])
                ]
        for (lower, upper) in boundaries:
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")

        mask = cv2.inRange(img, lower, upper)
        output = cv2.bitwise_and(img, img, mask=mask)
        cv2.imshow("output", output)



cv2.namedWindow('colorpickwindow')
cv2.setMouseCallback('colorpickwindow', colorpickwindow)

cv2.imshow('colorpickwindow', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

```










