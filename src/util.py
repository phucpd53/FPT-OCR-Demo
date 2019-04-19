import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DEBUG = True
debug_folder = "debug"
os.makedirs(debug_folder, exist_ok=True)

def is_table_header(line, headers):
    '''
    check if line include some key word of headers or not
    '''
    for header in headers:
        if header in "".join([item[0] for item in line]):
            return True
    return False

def is_next(previous, current, is_phrase=True):
    '''
    check if 2 input texts (previous, current) should be grouped together or not
    if 2 texts are close to each other then it should be treated as same text
    there are 2 situation:
    - input are 2 phrases then check the top down position
    - input are 2 words then check the left right position
    '''
    # in case arg is phrase type
    if is_phrase:
        # parameters
        space_param = 3
        ratio_param = 0.5
        
        # if the space between 2 word are too far
        if current['left'] - previous['left'] > space_param * previous['width']:
            return False
        # if height of 2 words are too different
        #if abs(current['height'] - previous['height']) / current['height'] > ratio_param:
        #    return False
        return True
    # in case arg is paragraph
    else:
        # parameters
        space_param = 2.0
        ratio_param = 1.0
        
        l_c, t_c, h_c, w_c = current[1]
        l_p, t_p, h_p, w_p = previous[1]
        # if the space between 2 phrase are too far
        if t_c - t_p > space_param * h_p:
            return False
        # if the position of 2 phrases are not aligned (left_aligned, middle_aligned, right_aligned)
        if min(abs(l_c - l_p), abs(l_c + w_c / 2 - l_p - w_p / 2), abs(l_c + w_c - l_p - w_p)) > ratio_param * h_p:
            return False
        return True
def remove_horizontal_line(img):
    # new_table_img = cv2.erode(table_img, np.ones((3,3), np.uint8), iterations=1)
    _, horizontal = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

    # get all horizontal line
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (200,1))
    horizontal = cv2.dilate(horizontal, kernel, iterations=1)

    # make horizontal line bigger
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (img.shape[1],1))
    horizontal = cv2.erode(horizontal, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,3))
    horizontal = cv2.erode(horizontal, kernel, iterations=1)


    horizontal_indx = horizontal == 0
    new_img = img.copy()
    new_img[horizontal_indx] = 255

    return new_img
def get_vertical_line(img):
    _, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    save(img, "threshold.jpg")
    # get all horizontal line
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,200))
    img = cv2.dilate(img, kernel, iterations=1)
    save(img, "vertical.jpg")
    vertical_x = []
    i = 0
    while i < img.shape[1]:
        # choose the row in the middle of image to check
        if img[img.shape[0]//2, i].all() == 0:
            vertical_x.append(i)
            i += img.shape[1] // 100
        else:
            i += 1
    return vertical_x

def save(img, name):
    if DEBUG:
        cv2.imwrite(os.path.join(debug_folder, name), img)
    
def df_to_image(df, img):
    font_path = "data/fonts-japanese-mincho.ttf"
    overlay = np.full(img.shape, 255, dtype=np.uint8)
    pil_image = Image.fromarray(overlay)
    draw = ImageDraw.Draw(pil_image)
    for index, row in df.iterrows():
        if row['conf'] != -1:
            # cv2.putText(overlay,row['text'],(row['left'],row['top'] + row['height']), cv2.FONT_HERSHEY_SIMPLEX, img.shape[0]/1000.0,0,1,cv2.LINE_AA)
            draw.text((row['left'],row['top'] + row['height']), row['text'], font=ImageFont.truetype(font_path, np.max(img.shape)//100), fill=0)
    return np.array(pil_image)