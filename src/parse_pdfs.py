
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

from pdfminer.high_level import extract_text_to_fp, extract_text

from tqdm import tqdm

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


    # tournaments = {x.split('-')[0]: x for x in listdir(join(base_directory, 'shots_xml'))}

    # for k, v in tournaments.items():
    #     cmd = f"cp {join(base_directory, 'shots_xml', v)} {join(base_directory, 'src', 'sample_xmls')}".replace(r"'", r"\'")
    #     system(cmd)
    
    # for fil in listdir(join(base_directory, 'src', 'sample_xmls')):
    #     remove_spans(fil)
        # print(fil)
        
    
    xmls = [x for x in listdir(join(base_directory, 'src', 'sample_xmls')) if x.endswith('.xml')]

    # This will be wrapped in a loop for each tournament
    
    # tree = et.parse(xmls[0])
    # tree = et.parse(xmls[1])
    # root = tree.getroot()

    # Shot Buffers
    shot_buffers = {'left': 20, 'right': 115, 'up': 240, 'down': 13}
    date_box = {'left': 0, 'right': 360, 'top': 80, 'bottom': 210}
    headers_box = {'top': 0, 'bottom': 80}
    sheet_pattern = re.compile(f'.*Sheet ([a-zA-Z0-9])')

    shots = []

    for doc in xmls:
        root = et.parse(join(base_directory, 'src', 'sample_xmls', doc)).getroot()
        print(doc)

        # TODO: This is where I can find the tournament and session information        
        start_time = ''
        date = ''
        sheet = ''
        title = ''
        header_parse = []

        for page in root:
            page_num = int(page.attrib['number'])
            banner_parse = []
            # print(page.tag, page.attrib)

            for item in page.findall('text'):

                # Check if the item is a team: player tag
                if (item.text is not None):
                    item_top = int(item.attrib['top'])
                    item_left = int(item.attrib['left'])
                    
                    # -- Parse for Shots --
                    if (': ' in item.text):
                        # These are all of the shots
                        shots.append([page_num, item_top, item_left])
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
                                    # Else do this
                                    throw_turn = element.text
                                    
                                # print(element.text)
                        shots[-1].extend([player, throw_num, throw_type, throw_rating, throw_turn])
                        # print('')
                    
                    # -- Parse for Score --
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
                
            # print(banner_parse)
            # print(f"Score: {team_1_score} : {team_2_score}")

            end_number = int(page.attrib['number'])
        
        # -- Document Parsing --
        # header_parse.sort(key=lambda x: len(x))
        header_pattern = re.compile(r'.* [0-9]{4}')
        for i in sorted(header_parse, key=len):
            if header_pattern.findall(i):
                title = i
            elif i == 'Curling':
                title 
                        

        print(start_time)
        print(date)
        print(sheet)
        print(title)
        print(header_parse)
                        
        print('\n')



    
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
        filename (_type_): The filename of the file to remove the spans from
    """
    with open(join(base_directory, 'shots_xml', filename), "r", encoding = "ISO-8859-1") as fh:
        xml_text = fh.read()
        xml_text = re.sub(r'</span>.*">', ' ', xml_text)

    with open(join(base_directory, 'shots_xml', filename), "w") as fh:
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
