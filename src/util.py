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