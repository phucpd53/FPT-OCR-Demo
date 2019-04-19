import numpy as np
import cv2
import json
import pytesseract
from pytesseract import Output
from PIL import Image
from src import separate_table, ocr

def main():
    path = "data/007.jpg"
    
    json_file = "content.json"
    
    img = cv2.imread(path)
    # TODO: it does not work
    # dealing with rotated image. Choose the best image's direction based on tesseract confidence score
    conf = -1 
    _img = img.copy()
    #################################################################################################
    for i in range(1):
        _img = (np.rot90(_img) if i != 0 else _img)
        _df = pytesseract.image_to_data(Image.fromarray(_img), lang="jpn", output_type=Output.DATAFRAME)
        _df.columns = _df.columns.str.strip()
        _df = _df.dropna()
        if conf <= _df['conf'].mean():
            conf = _df['conf'].mean()
            img = _img
    print("Divide table and metadata...")
    meta_img, table_img = separate_table.run(img)
    
    print("Process metadata...")
    # process meta data using tesseract ocr
    all_blocks = ocr.process_meta(meta_img)
    # tagging each block into specific content using regular expression
    tags = ocr.tag_paragraph(all_blocks)
    tags['from'] = ocr.tag_detail([tags['from']])
    tags['to'] = ocr.tag_detail([tags['to']] + tags['unknown'])
    del tags['unknown']
    
    print("Process table...")
    # process table data using tesseract ocr
    tags['table'] = ocr.process_table(table_img)
    
    # write to json
    with open(json_file, "w", encoding='utf8') as f:
        json.dump(tags, f, ensure_ascii=False, indent=4)

main()