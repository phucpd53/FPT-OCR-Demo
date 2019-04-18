import cv2 
import numpy as np
from matplotlib import pyplot as plt

import skimage
import imutils
from imutils import contours
from tqdm import tqdm
import pandas as pd

def run(img):
    kernel = np.ones((5,5),np.uint8)
    img_erode = cv2.erode(img,kernel,iterations = 3)

    out_e = cv2.cvtColor(img_erode, cv2.COLOR_BGR2GRAY)
    ret, out_e_thresh_bin = cv2.threshold(out_e, 120, 255, cv2.THRESH_BINARY_INV)
    out_e_thresh_bin_nonoise = cv2.morphologyEx(out_e_thresh_bin,cv2.MORPH_OPEN,kernel, iterations = 2)
    out_e_thresh_bin_e = cv2.erode(out_e_thresh_bin_nonoise, None, iterations=4)

    labels = skimage.measure.label(out_e_thresh_bin_e, neighbors=8, background=0)
    mask = np.zeros(out_e_thresh_bin_e.shape, dtype="uint8")

    img_size = img.shape[0]*img.shape[1]
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue

        # otherwise, construct the label mask and count the
        # number of pixels 
        labelMask = np.zeros(out_e_thresh_bin_e.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)

        # if the number of pixels in the component is sufficiently
        # large, then add it to our mask of "large blobs"
        if numPixels > img_size*1/1000:
            cnts_mask_all = cv2.findContours(labelMask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cnts_mask = grab_contours(cnts_mask_all)
            cnts_mask = contours.sort_contours(cnts_mask)[0]
            contour_size = []
            for i, c in enumerate(cnts_mask):
                (x,y,w,h) = cv2.boundingRect(c)
                contour_size.append((x+w)*(y+h))
            if np.max(contour_size) <= img_size*90/100:
                mask = cv2.add(mask, labelMask)
        else:
            continue
    cnts_allinfo = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    cnts = grab_contours(cnts_allinfo)
    cnts = contours.sort_contours(cnts)[0]

    counter = 0
    img_dict = {}
    table_part = np.zeros((1, 1), np.uint8)
    other_part = np.zeros((1, 1), np.uint8)
    for (i, c) in enumerate(cnts):
        #draw bright spots on image
        (x, y ,w, h) = cv2.boundingRect(c)
        img_crop = img.copy()[y:y+h,x:x+w,:]
        if table_part.shape[0] * table_part.shape[1] <= w * h:
            table_part = img_crop
            other_part = img[0:y, :, :]
    return other_part, table_part

"""Connected-component analysis of image"""

def grab_contours(cnts):
    # if the length the contours tuple returned by cv2.findContours
    # is '2' then we are using either OpenCV v2.4, v4-beta, or
    # v4-official
    if len(cnts) == 2:
        cnts = cnts[0]

    # if the length of the contours tuple is '3' then we are using
    # either OpenCV v3, v4-pre, or v4-alpha
    elif len(cnts) == 3:
        cnts = cnts[1]

    # otherwise OpenCV has changed their cv2.findContours return
    # signature yet again and I have no idea WTH is going on
    else:
        raise Exception(("Contours tuple must have length 2 or 3, "
            "otherwise OpenCV changed their cv2.findContours return "
            "signature yet again. Refer to OpenCV's documentation "
            "in that case"))

    # return the actual contours array
    return cnts