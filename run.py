import cv2
import json
from src import separate_table, ocr

def main():
    path = "data/005.jpg"
    
    json_file = "content.json"
    
    img = cv2.imread(path)
    
    meta_img, table_img = separate_table.run(img)
    
    # process meta data using tesseract ocr
    all_blocks = ocr.process_meta(meta_img)
    # tagging each block into specific content using regular expression
    tags = ocr.tag_paragraph(all_blocks)
    tags['from'] = ocr.tag_detail([tags['from']])
    tags['to'] = ocr.tag_detail([tags['to']] + tags['unknown'])
    
    # write to json
    del tags['unknown']
    with open(json_file, "w", encoding='utf8') as f:
        json.dump(tags, f, ensure_ascii=False, indent=4)

main()