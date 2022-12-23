
import logging
from os import mkdir, listdir, system
from os.path import isdir, isfile, join
from pathlib import Path
import re
from time import sleep
from typing import Optional

import xml.etree.ElementTree as et

from borb.pdf import Document
from borb.pdf import PDF
from borb.toolkit import SimpleTextExtraction

from PyPDF2 import PdfReader

from pdfminer.high_level import extract_text_to_fp, extract_text, extract_pages
from pdfminer.layout import LTLine, LTCurve

import fitz

import numpy as np
import pandas as pd
import plotnine as gg
from tqdm import tqdm

import cv2

from CurlingDB import CurlingDB


base_directory = str(Path(__file__).parent.parent.resolve())

# Logging
# Create formatter and the two handlers
log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s - %(name)s')
f_handler = logging.FileHandler('log.log')
s_handler = logging.StreamHandler()

# Set the levels and formats of the handlers
f_handler.setLevel(logging.DEBUG)
s_handler.setLevel(logging.INFO)
f_handler.setFormatter(log_format)
s_handler.setFormatter(log_format)

# Get the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(f_handler)
logger.addHandler(s_handler)


def main():

    # Create/connect to database
    # db = CurlingDB()
    # db.drop_tables()
    # db.create_tables()

    tournaments = {x.split('-')[0]: x for x in listdir(join(base_directory, 'shots_xml'))}

    # doc = 'CUR_1819_CWC_4P-Men\'s_Teams-03~Session_3-NOR-SCO.pdf'
    doc = 'CUR_1819_CWC_4P-Men\'s_Teams-03~Session_3-NOR-SCO.xml'
    doc = 'CU_WMCC2016P-Men\'s_Teams-01~Session_1-GER-SUI.pdf'
    # pdf_to_img(doc, 8)
    # quit()

    # xmls = [x for x in listdir(join(base_directory, 'src', 'sample_xmls')) if x.endswith('.xml')]


    # --- Stone Detection ---
    # stones = pd.concat([get_stone_positions(filename=join(base_directory, 'src', 'sample_images', f'page_{page_num}.png')) for page_num in range(1, 9)])

    # print(stones.shape)
    # print(stones.info())
    # print(stones.describe())

    # g = gg.ggplot(stones, gg.aes(x='x', y='-y', colour='factor(stone_colour)')) + gg.geom_point()
    # print(g)

    # --- Text Parsing ---
    title, doc_info, page_df, shot_df = parse_text(join(base_directory, 'src', 'sample_xmls', list(tournaments.values())[1]))
    print(title)
    print(doc_info)
    print(page_df)
    print(shot_df.info())

    # --- Steps ---
    # Take a pdf file 
    # Convert the pdf to xml for text parsing - complete
    # Run the xml through the span cleaning - complete
    # Convert the pdf to images for stone parsing - complete
    # Parse the xml for text - complete
    # Parse the images for the stone locations - Need to identify house location
    # Delete the xml and images - can do, TODO
    # Enter into database - TODO
    
    

    
    # cv2.destroyAllWindows()
    
    
    # Profit?

def get_stone_positions(filename: str) -> pd.DataFrame:
    """Take a full path to a file and return a dataframe with the stone positions.

    Args:
        filename (str): Full path to image to read.

    Returns:
        pd.DataFrame: Dataframe with columns [colour, x, y] containing all of the full-size stones.
    """

    # This is not 100%, there are some complicated things going on here.
    # Weaknesses include finding stones that are slightly overlapping with locations of previous stones, many of these are ignored
    # - This can be adjusted by the continuation thresholds

    lower_red = np.array([0, 250, 250])
    upper_red = np.array([0, 255, 255])
    lower_yellow = np.array([28, 200, 200])
    upper_yellow = np.array([32, 255, 255])
    lower_blue = np.array([120, 225, 225])
    upper_blue = np.array([130, 255, 255])

    # Convert the image into a cv2 BGR object
    frame = cv2.imread(filename)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create the masks
    red_mask = cv2.inRange(hsv, lower_red, upper_red)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    yellow_mask = cv2.bitwise_or(yellow_mask, blue_mask)

    stones = []

    for stone_colour, colour_mask in {'red': red_mask, 'yellow': yellow_mask}.items():
        # Find contours and bounding boxes
        contours, _ = cv2.findContours(colour_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        con_bound = [cv2.boundingRect(con) for con in contours]

        # For enumerate contours, if contour centre is inside only one bounding box, then add it to good list
        good_con = []
        for i, con in enumerate(contours):
            # If bounding box is too small or large, skip
            if min(con_bound[i][2:]) < 25: continue
            elif max(con_bound[i][2:]) > 50: continue
            elif (len(con) < 40) or (len(con) > 75): continue

            # If the centre of the contour is within the bounding box of another, skip
            # This is to avoid the outlines of previous stones
            x, y, w, h = con_bound[i]
            x, y = (x + w/2, y + h/2)
            count = 0
            for x_o, y_o, w, h in con_bound:
                if (x > x_o) and (x < x_o + w) and (y > y_o) and (y < y_o + h):
                    count += 1
                    if count > 1:
                        break
            else:
                good_con.append(con)
            

        # Find the centres of the rocks
        # This is an averaging of the bounding box and mean
        for con in good_con:
            con_unwrap = [x[0] for x in con]
            centroid_x = round(sum([x[0] for x in con_unwrap])/len(con_unwrap))
            centroid_y = round(sum([x[1] for x in con_unwrap])/len(con_unwrap)) 

            x, y, w, h = cv2.boundingRect(con)

            avg_x = round((centroid_x + x + w/2)/2)
            avg_y = round((centroid_y + y + h/2)/2)
        
            stones.append((stone_colour, avg_x, avg_y))
    
    logger.info(f"Completed stone parsing page {filename.split('/')[-1]}")
    return pd.DataFrame(stones, columns=['stone_colour', 'x', 'y'])

    
def pdf_to_img(filename: str, zoom: int=8):
    """Take a file name and convert all pages to images, saved in the sample_images directory

    Args:
        filename (str): filename of pdf to convert to images
    """

    doc = fitz.open(join(base_directory, 'src', filename))
    # Parse the pdf into images
    for i, page in enumerate(doc.pages()):
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        pix.save(join(base_directory, 'src', 'sample_images', f"page_{i + 1}.png"))
        logger.info(f"Completed {i}")

        
def parse_text(filename) -> tuple[str, tuple, pd.DataFrame, pd.DataFrame]:
    xmls = [x for x in listdir(join(base_directory, 'src', 'sample_xmls')) if x.endswith('.xml')]

    # Zones and buffers
    shot_buffers = {'left': 20, 'right': 115, 'up': 240, 'down': 13}
    date_box = {'left': 0, 'right': 360, 'top': 80, 'bottom': 210}
    headers_box = {'top': 0, 'bottom': 80}
    sheet_pattern = re.compile(f'.*Sheet ([a-zA-Z0-9])')

    # Lists to be created with appends
    page_info = []
    shot_info = []

    # Get the root of the xml file
    root = et.parse(filename).getroot()

    # TODO: This is where I can find the tournament and session information        
    # Initialize the per-document fields, some of them might not parse correctly
    start_time = ''
    date = ''
    sheet = ''
    title = ''
    header_parse = []

    # For each page of the document
    for page in root:
        page_num = int(page.attrib['number'])
        banner_parse = []

        # For each text element of the page
        for item in page.findall('text'):

            # Check if the item is a team: player tag
            if (item.text is not None):
                item_top = int(item.attrib['top'])
                item_left = int(item.attrib['left'])
                
                # -- Parse for Shots --
                if (': ' in item.text):
                    # These are all of the shots
                    shot_info.append([page_num, item_top, item_left])
                    player = throw_num = throw_type = throw_rating = throw_turn =  None
                    for element in page.findall('text'):
                        # For each element, check if it is in the zone of the name label
                        element_top = int(element.attrib['top'])
                        element_left = int(element.attrib['left'])
                        if (((element_top - item_top) <= shot_buffers['down']) and ((element_top - item_top) >= -shot_buffers['up'])
                        and ((element_left - item_left) <= shot_buffers['right']) and ((element_left - item_left) >= -shot_buffers['left'])):
                            # If the difference is zero, set the name
                            if (element_top == item_top) and (element_left == item_left):
                                # If the element is the player
                                player = element.text
                            elif (element_top - item_top) + 10 < 0:
                                # If the location is far above
                                throw_num = element.text
                            elif (abs(element_left - item_left) < 2) and (element_top > item_top):
                                # If the element is directly below
                                throw_type = element.text
                            elif (element.text is not None) and (('%' in element.text) or (len(element.text) < 2)):
                                # If the element contains a '%' or is short like the single number ratings
                                throw_rating = element.text
                            else:
                                # Else it is the throw turn direction
                                throw_turn = element.text
                                
                    shot_info[-1].extend([player, throw_num, throw_type, throw_rating, throw_turn])
                
                # -- Parse for End Score --
                # This is for the more common string case
                if ('=' in item.text) and (abs(item_top - 225) < 5):
                    banner_parse.append((int(item.attrib['top']), int(item.attrib['left']), int(item.text.split()[-1].strip(' ='))))
                
                # Less common split case
                if (abs(item_top - 225 < 5)) and (len(item.text.strip()) < 3) and (len(item.text.strip()) > 0) and all(x not in item.text for x in ['+', '-']):
                    banner_parse.append((int(item.attrib['top']), int(item.attrib['left']), int(item.text.strip())))
                
                # -- Parse for Title Info --
                # Only check first page
                if int(page.attrib['number']) == 1:
                    # Check for the start time
                    if 'Start Time' in item.text:
                        start_time = item.text.strip().split()[-1]
                    # Check for the date
                    if (len(item.text.strip().split()) == 4) and (item_top > date_box['top']) and (item_top < date_box['bottom']) and (item_left < date_box['right']):
                        date = item.text.strip()
                    # Check for a sheet label
                    if (item_top < 200) and ('Sheet' in item.text):
                        sheet = sheet_pattern.findall(item.text)[0]
                        
                    # Check for the rest of the headers
                    if (item_top < headers_box['bottom']):
                        header_parse.append(item.text)
                        
        # -- Page Parsing --
        # Each page will have either 
        if len(banner_parse) == 2:
            team_1_score = banner_parse[0][2]
            team_2_score = banner_parse[1][2]
        elif len(banner_parse) == 4:
            team_1_score = banner_parse[0][2] + banner_parse[2][2]
            team_2_score = banner_parse[1][2] + banner_parse[3][2]
        else:
            team_1_score = 99
            team_2_score = 99
            logger.warning(f"Found invalid set of score elements")
            
        end_number = page_num

        logger.info(f"Completed text parsing of page {filename.split('/')[-1]}")
        page_info.append((end_number, team_1_score, team_2_score))
    

    # -- Document Parsing --
    header_pattern = re.compile(r'.* [0-9]{4}')
    for i in sorted(header_parse, key=len):
        if header_pattern.findall(i):
            title = i
        elif i == 'Curling':
            title 
    
    tourney_info = title
    doc_info = (date, start_time, sheet)

    page_df = pd.DataFrame(page_info, columns=['end_number', 'team_1_score', 'team_2_score'])
    shot_df = pd.DataFrame(shot_info, columns=['end_number', 'top', 'left', 'player', 'throw_num', 'throw_type', 'throw_rating', 'throw_turn'])

    return tourney_info, doc_info, page_df, shot_df
                    
                    



    
    # Have to look for 'prepositioned stones'
    # for i in shots:
    #     print(i)



    
    # I think that the winner is pdftohtml for the metadata
    # Can just convert each of the files and take stuff by location
    # Find the player name of each and build a box around it to find the number, throw type, score and throw number

    
    # First line is typically the tournament name followed by a location


def remove_spans(filename):
    """There are some weird spans introduced in the parsing but this will remove them in place.

    Args:
        filename (str): The full filename of the file to remove the spans from
    """
    with open(filename) as fh:
        xml_text = fh.read()
        xml_text = re.sub(r'</span>.*">', ' ', xml_text)

    with open(filename, "w") as fh:
        fh.write(xml_text)

def pdf_to_html():
    """Cycle through pdfs and create xmls using pdftohtml with the xml option."""

    if not isdir(join(base_directory, 'shots_xml')):
        mkdir(join(base_directory, 'shots_xml'))

    pdfs = [x.strip().replace(r"'", r"\'") for x in listdir(join(base_directory, 'shots'))]
    for pdf in tqdm(pdfs):
        cmd = f"pdftohtml -xml {join(base_directory, 'shots', pdf).strip()} {join(base_directory, 'shots_xml', pdf[:-4])}"
        print(cmd)
        i = system(cmd)


if __name__ == '__main__':
    main()
