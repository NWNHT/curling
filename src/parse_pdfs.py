
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


    # tournaments = {x.split('-')[0]: x for x in listdir(join(base_directory, 'shots'))}

    # for k, v in tournaments.items():
    #     print(k, v)
    
    xmls = [x for x in listdir(join(base_directory, 'src')) if x.endswith('.xml')]

    # tree = et.parse(xmls[0])
    # tree = et.parse(xmls[1])
    # root = tree.getroot()
    shots = []
    left_buf = 20
    right_buf = 115 # orig 94
    up_buf = 240 # orig 234
    down_buf = 13
    for doc in xmls:
        root = et.parse(doc).getroot()

        # TODO: This is where I can find the tournament and session information        
        
        

        for page in root:
            page_num = int(page.attrib['number'])
            print(page.tag, page.attrib)

            for item in page.findall('text'):
            # for item in page:
                if (item.text is not None):
                    if (': ' in item.text):
                        # These are all of the shots
                        item_top = int(item.attrib['top'])
                        item_left = int(item.attrib['left'])
                        shots.append([page_num, item_top, item_left])
                        player = throw_num = throw_type = throw_rating = throw_turn =  None
                        for element in page.findall('text'):
                            # For each element, check if it is in the zone of the name label
                            element_top = int(element.attrib['top'])
                            element_left = int(element.attrib['left'])
                            if (((element_top - item_top) <= down_buf) and ((element_top - item_top) >= -up_buf)
                            and ((element_left - item_left) <= right_buf) and ((element_left - item_left) >= -left_buf)):
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
                                    
                                print(element.text)
                        shots[-1].extend([player, throw_num, throw_type, throw_rating, throw_turn])
                        print('')
    
    # Have to look for 'prepositioned stones'
    for i in shots:
        print(i)



    
    # I think that the winner is pdftohtml for the metadata
    # Can just convert each of the files and take stuff by location
    # Find the player name of each and build a box around it to find the number, throw type, score and throw number

    
    # First line is typically the tournament name followed by a location


def remove_spans(filename):
    """There are some weird spans introduced in the parsing but this will remove them in place.

    Args:
        filename (_type_): The filename of the file to remove the spans from
    """
    with open(join(base_directory, 'shots_xml', filename), "r") as fh:
        xml_text = fh.read()
        xml_text = re.sub(r'<\/span>.*ft1\">', ' ', xml_text)

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
