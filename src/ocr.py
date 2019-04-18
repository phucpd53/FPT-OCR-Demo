import pytesseract
from pytesseract import Output
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from glob import glob
import pandas as pd
import os
import re
from src import util

# REGULAR EXPRESSION template
################################################################
post_number = r"〒\d{3}-\d{4}"
address = r".*(都|道|府|県|区|市|町|丁).*"
building = r".*(ビル|マンション|ハイツ|アパート).*"
phone_number = r".*0\d*-\d*-\d*"
money_number = r".*(v|\\)\d*\.\d*"
date = r".*\d{2,4}年\d{1,2}月\d{1,2}日"
# date = r".*\d{2,4}(年|\/|-)\d{1,2}(月|\/|-)\d{1,2}"
dest = r".*御中"
organize = r".*株式会社.*"
title = r"ご..書"
################################################################

def process_meta(img):
    table_header = ['ご注文日', '商品ID', '商品名', '数量', '単位', '単価', '小計']
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.blur(img, ksize=(3, 3))
    _,img = cv2.threshold(img,200,255,cv2.THRESH_BINARY)
    # dealing with rotated image. Choose the best image's direction based on tesseract confidence score
    df = None
    conf = -1 
    _img = None
    for i in range(4):
        img = (np.rot90(img) if i != 0 else img)
        _df = pytesseract.image_to_data(Image.fromarray(img), lang="jpn", output_type=Output.DATAFRAME)
        _df.columns = _df.columns.str.strip()
        _df = _df.dropna()
        if conf < _df['conf'].mean():
            conf = _df['conf'].mean()
            df = _df
            _img = img.copy()

    df.columns.str.strip()
    df = df.dropna()
    df = df.drop(columns=['level', 'page_num'])

    # group text from tesseract's result from dataframe format into block
    all_blocks = []
    for _, block in df.groupby('block_num'):
        str_block = []
        for _, parse in block.groupby('par_num'):
            for _, line in parse.groupby('line_num'):
                list_lines = df_to_list(line)
                left, top, height, width = list_lines[0][1]
                str_block.append(list_lines) 
        all_blocks.append(str_block)
    return all_blocks


def row_to_line(rows):
    '''
    merge multiple texts with position into one line and its position
    '''
    line = ""
    left, top, height, width = (9999,9999,0,0) 
    for row in rows:
        line += row['text']
        left = min(left, row['left']) # Left of a line of characters should be the character on the left
        top = min(top, row['top'])    # Top of a line of characters should be the the min top of all characters (each character's top is not the same)
        height = max(height, row['top'] + row['height'] - top)
        width = max(width, row['left'] + row['width'] - left)
    return (line, (left, top, height, width))

def df_to_list(df):
    '''
    @input: a dataframe
    divide a data frame of line into different elements(blocks) 
    if these blocks locate far from eachother
    @output: a list of tupple (phrase, location)
    '''
    # result will be saved here
    phrases = []
    
    # initialize for for_loop
    previous_row = df.iloc[0]
    rows = [df.iloc[0]]
    
    for indx in range(1, df['text'].count()):
        row = df.iloc[indx]
        if util.is_next(previous_row, row):
            rows.append(row)
        else:
            previous_item = None
            text, location = row_to_line(rows)
            phrases.append((text, location))
            rows = [row]
        previous_row = row
    if len(rows) != 0:
        text, location = row_to_line(rows)
        phrases.append((text, location))
    return phrases

def tag_paragraph(blocks):
    '''
    find the meaning of each paragraph by regular expression
    label paragraph using tag
    '''
    tags = {}
    tags['from'] = ""
    tags['to'] = ""
    tags['total_bill'] = ""
    tags['date'] = ""
    tags['title'] = ""
    tags['unknown'] = []
    for block in blocks:
        for idx, lines in enumerate(block):
            line = "".join([line[0] for line in lines])
            if re.match(organize, line):
                if re.match(dest, line):
                    tags["to"] = block
                    continue
                else: 
                    tags["from"] = block
                    continue
            elif re.match(money_number, line):
                tags["total_bill"] = re.findall(r"\\.*", line)[0]
                continue
            elif re.match(date, line):
                tags["date"] = re.findall(r"\d{2,4}.\d{1,2}.\d{1,2}.", line)[0]
                continue
            elif re.match(title, line):
                tags["title"] = line
                continue
            if idx == len(block) - 1:
                tags['unknown'].append(block)
    return tags

def tag_detail(blocks):
    _dict = {}
    for block in blocks:
        for lines in block:
            for line, _ in lines:
                if re.match(post_number, line):
                    _dict["post"] = line
                if re.match(address, line):
                    _dict["address"] = line
                if re.match(building, line):
                    _dict["building"] = line
                if re.match(phone_number, line):
                    _dict["TEL"] = line
                if re.match(organize, line):
                    _dict["ORG"] = line
    return _dict