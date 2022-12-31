
import datetime
import logging
from os import mkdir, listdir, system, remove
from os.path import isdir, isfile, join
from pathlib import Path
import re
from time import sleep
from typing import Optional

import xml.etree.ElementTree as et

# from borb.pdf import Document
# from borb.pdf import PDF
# from borb.toolkit import SimpleTextExtraction

from PyPDF2 import PdfReader

from pdfminer.high_level import extract_text_to_fp, extract_text, extract_pages
from pdfminer.layout import LTLine, LTCurve

import fitz

import dateparser
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

    # --- Create/connect to database ---
    db = CurlingDB(db_name='order_testing.db')
    
    parsed_tournaments_file = 'parsed_tournaments.txt'
    parsed_tournaments_path = join(base_directory, 'src', parsed_tournaments_file)
    reset = True # If true this will reset the database and delete the parse_tournament document
    if reset:
        logger.info(f"Reseting database and parsed tournament file.")
        db.drop_tables()
        db.create_tables()
        if isfile(parsed_tournaments_path):
            remove(parsed_tournaments_path)
    
    # Conversion
    pixel_to_m = 0.007370759

    # pdf_dir = join(base_directory, 'src', 'sample_pdfs')
    pdf_dir = join(base_directory, 'src', 'tourney_sample_pdfs')
    # pdf_dir = join(base_directory, 'src', 'small_tourney_sample_pdfs')
    # pdf_dir = join(base_directory, 'src', 'alt_sample_pdfs')
    # pdf_dir = join(base_directory, 'shots')
    temp_dir = join(base_directory, 'src', 'temp')
    

    # Find all tournaments in directory
    tournaments = {x.split('-')[0]: x for x in listdir(pdf_dir) if x.endswith('pdf')}.keys()

    # Find all tournaments already parsed
    if isfile(parsed_tournaments_path):
        logger.info(f"Parsed file already exists.")
        with open(parsed_tournaments_path, 'r') as fh:
            parsed_tournaments = [x.strip() for x in fh.readlines()]
        # Find all tournaments in pdf dir and not already parsed
        tournaments = list(set(tournaments).symmetric_difference(set(parsed_tournaments)))
    else:
        # Make an empty file
        logger.info(f"Parsed file does not exist, creating.")
        open(parsed_tournaments_path, 'w')
    
    logger.info(f"Looping for tournaments {tournaments}")

    for tournament in tournaments:
        tournament_name = ''

        pdfs = [pdf for pdf in listdir(pdf_dir) if (pdf.startswith(tournament)) and (pdf.endswith('pdf'))]

        # Blacklist some problem documents
        # The first two are forfeits that aren't properly parsed by this and the last pdf is a corrupted pdf from the source
        problem_pdfs = ['CUR_WMDCC2019P-Mixed_Doubles-26~Session_26-SVK-BEL.pdf', 'CUR_WMDCC2017P-Mixed_Doubles-27~Session_27-WAL-BUL.pdf', 'CU_WMCC2016P-Mens_Teams-07~Session_7-SCO-CAN.pdf']
        for i in problem_pdfs:
            try:
                pdfs.remove(i)
            except:
                pass

        logger.info(f"Found {len(pdfs)} pdfs for tournament {tournament}.")
        tournament_info = []
        for pdf in pdfs:
            # --- Steps ---
            # Take a pdf file 
            # Convert the pdf to xml for text parsing - complete
            # Run the xml through the span cleaning - complete
            # Convert the pdf to images for stone parsing - complete
            # Parse the xml for text - complete
            # Parse the images for the stone locations - complete
            # Delete the xml and images - complete
            # Enter into database - complete
            
            logger.info(f"Beginning pdf: {pdf}")

            # -- Take team name info from pdf title --
            team1, team2 = [name.split('.')[0] for name in pdf.split('-')[-2:]]
            game_type = pdf.split('-')[1]
            
            # -- Convert pdf to xml --
            # This sometimes fails a single time but it loses progress for the entire tournament
            #   This will just try again a single time if the pdf converion fails
            try:
                pdf = pdf.replace(r"'", r"\'").strip()
                pdf_to_xml(pdf, pdf_dir=pdf_dir, temp_dir=temp_dir)

                xml = [xml for xml in listdir(temp_dir) if xml.endswith('xml')][0]
            except:
                logger.warning(f"Failed to convert pdf to xml, document: {pdf}")
                pdf_to_xml(pdf, pdf_dir=pdf_dir, temp_dir=temp_dir)

                xml = [xml for xml in listdir(temp_dir) if xml.endswith('xml')][0]
                

            # Remove the erroneous <span> elements
            remove_spans(xml, temp_dir=temp_dir)

            # -- Convert pdf to images --
            pdf_to_img(pdf, zoom=8, pdf_dir=pdf_dir, temp_dir=temp_dir)

            # -- Parse xml --
            tourney_info, doc_info, page_df, shot_df = parse_text(xml, temp_dir=temp_dir)

            if tournament_name == '':
                tournament_name = tourney_info

            # -- Parse images --
            image_files = [img_ for img_ in listdir(temp_dir) if img_.startswith('page')]
            all_stones = []
            hammers = []
            dir_of_play_list = []
            num_positions_list = []
            for i, img_filename in enumerate(image_files):
                i = int(img_filename.split('_')[1].split('.')[0])
                stones, hammer, dir_of_play, num_positions = get_stone_positions(join(temp_dir, img_filename), i)
                all_stones.append(stones)
                hammers.append(hammer)
                dir_of_play_list.append(dir_of_play)
                num_positions_list.append(num_positions)
            
            all_stones = pd.concat(all_stones)

            # -- Remove temporary files --
            for temp_file in listdir(temp_dir):
                remove(join(temp_dir, temp_file))
            
            print('\n\n')
            tournament_info.append((tournament, game_type, team1, team2, doc_info, page_df, shot_df, all_stones, hammers, dir_of_play_list, num_positions_list))
        
        # --- Organize Data ---
        # doc_info: (start_time, sheet, team_1_score_final, team_2_score_final)
        # page_df: (end_number, team_1_score, team_2_score)
        # shot_df: (end_number, x_ind, y_ind, player, throw_num, throw_type, throw_rating, throw_turn)
        # stones_df: end, frame_num, frame_x, frame_y, stone_colour, x, y, stone_size
        # hammers: list of 'red' or 'yellow' representing the hammer for each end in a document
        # dir_of_play_down_list: list of bools representing if the play direction was down
        # num_positions_list: list of ints of the highest index of a frame on the page
        # houses_list: (end_num, ind_x, ind_y, orig_x, orig_y) - Not present

        # Pass the db, tournament info, 

        tournament_dates = [x.date() for x in [pdf[4][0] for pdf in tournament_info] if (x is not None) and (not isinstance(x, str))]
        tournament_start_date = min(tournament_dates)
        tournament_end_date = max(tournament_dates)
        event_data = (tournament_name, tournament, tournament_start_date, tournament_end_date, '')

        # Entry for the tournament/event
        tournament_cmd = """
        INSERT OR IGNORE INTO Event (name, abbrev, start_date, end_date, location)
                             VALUES (?, ?, ?, ?, ?)"""
        db.execute_command(tournament_cmd, event_data)
        tournament_id = db.cursor.lastrowid
        logger.debug(f"The tourn_id is: {tournament_id}")

        for match in tournament_info:
            # Entry for the match
            # Tournament abbrev, start_time, game type, sheet, team1, team1score, team2, team2score
            match_data = (match[4][0], match[1], match[4][1], match[2], int(match[4][2]), match[3], int(match[4][3]), tournament_id)
            match_cmd = """
            INSERT OR IGNORE INTO Match (start_time, type, sheet, team_1, team_1_final_score, team_2, team_2_final_score, event_id)
                                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)"""
            db.execute_command(match_cmd, match_data)
            match_id = db.cursor.lastrowid
            logger.debug(f"\tThe match_id is: {match_id}")

            # Set the sex of the match for later use
            if match[1] == 'Mens_Teams':
                sex = 'm'
            elif match[1] == 'Womens_Teams':
                sex = 'f'
            else:
                sex = 'u' # Unknown

            # Condition the dataframes for the ends, shots, and stone entries
            end_df = match[5]
            logger.info(f"end_df: {len(end_df)} hammers: {len(match[9])}")
            end_df['hammer'] = match[8]
            end_df['direction'] = match[9]
            end_df['num_positions'] = match[10]

            shot_df: pd.DataFrame = match[6]
            shot_df = shot_df.sort_values(by=['ind_y', 'ind_x']).reset_index(drop=True)

            stones_df = match[7]

            # For each end from the match
            for _, row in end_df.iterrows():
                # Make the end entry
                end_num = row['end_number']
                end_data = (end_num, row['hammer'], 'down' if row['direction'] else 'up', row['team_1_score'], row['team_2_score'], match_id)
                end_cmd = """
                INSERT OR IGNORE INTO End (num, hammer_colour, direction, team_1_final_score, team_2_final_score, match_id)
                                   VALUES (?, ?, ?, ?, ?, ?)"""
                db.execute_command(end_cmd, end_data)
                end_id = db.cursor.lastrowid
                logger.debug(f"\t\tThe end_id is: {end_id}")

                # Entries for all of the throws, also handles the players
                throw_colour = 'red' if row['hammer'] == 'yellow' else 'yellow'
                for index, shot_row in (shot_df.query('end_number == @end_num')
                                               .sort_values(by=['ind_y', 'ind_x'])
                                               .reset_index(drop=True)
                                               .iterrows()):
                    logger.debug(f"\t\t\tFrame: {shot_row['ind_x']} {shot_row['ind_y']}")

                    # Create the player entry
                    # If the player name and nationality already exists in the database, use that id
                    # - If sex in m/f then check if the player has a sex in the database, if not then set it
                    # If no entry then make an entry, if sex in m/f then insert that

                    player_team, player_name = [x.strip() for x in shot_row['player'].split(':')]
                    player_data = (player_name, player_team)
                    player_query = """
                    SELECT player_id, sex
                    FROM Player
                    WHERE name = ?
                      AND team = ?"""
                    db.execute_query(player_query, player_data)
                    player_response = db.cursor.fetchall()

                    if len(player_response) > 0:
                        # There is already a player
                        result = player_response[0]
                        # If the player has no sex but we do know it
                        if (result[1] == 'u') and (sex != 'u'):
                            logger.debug(f"There is a player {player_name} but no sex, we now know it.")
                            player_update_cmd = """
                            UPDATE Player
                            SET sex = ?
                            WHERE player_id = ?"""
                            db.execute_command(player_update_cmd, (sex, result[0]))
                        else:
                            logger.debug(f"There is a player {player_name} but no sex, we still don't know it.")
                    else:
                        # There is no player, add
                        logger.debug(f"There is no player {player_name}, we know the sex.")
                        player_add_cmd = """
                        INSERT OR IGNORE INTO Player (name, sex, team)
                                            VALUES (?, ?, ?)"""
                        db.execute_command(player_add_cmd, (player_name, sex, player_team))
                    
                    # By here, there should be a player that can be found in the next query

                    # Create a throw entry
                    throw_data = (index + 1, throw_colour, shot_row['throw_type'], shot_row['throw_rating'], player_name, player_team, end_id)
                    throw_cmd = """
                    INSERT OR IGNORE INTO Throw (throw_num, colour, type, rating, player_id, end_id)
                                         VALUES (?, ?, ?, ?, (SELECT player_id
                                                             FROM Player
                                                             WHERE name=?
                                                               AND team=?), ?)"""
                    db.execute_command(throw_cmd, throw_data)
                    throw_colour = 'red' if throw_colour == 'yellow' else 'yellow'
                
                # Filter to only the end of interest
                stones_df_end = stones_df.query('end == @end_num')
                # For each unique combination of frame_x and frame_y, sorted by y then x
                frames = sorted(stones_df_end.groupby(['frame_y', 'frame_x']).size().index)
                
                # For most of the mixed doubles there is a valid starting frame, for all mens/womens 
                #   and the single mixed doubles we have to make something up
                # If mens or womens, or if tournament == CU_WMDCC2016P
                # - Then make a position with no stones with frame number 0, then start on frame 1
                # Else: (All mixed doubles except that one)
                # - Make frames/positions beginning at zero
                # if (sex == 'm') or (sex == 'f') or (tournament == 'CU_WMDCC2016P'):
                #     # Create a single empty frame to act as a starting position
                #     position_data = (0, end_id)
                #     position_cmd = """
                #     INSERT OR IGNORE INTO Position (frame_num, end_id)
                #                             VALUES (?, ?)"""
                #     db.execute_command(position_cmd, position_data)

                #     # Set offset
                #     frame_offset = 1
                # else:
                #     # For the other cases there is already a starting position
                #     frame_offset = 0

                # For regular mens/womens ends the numbering of the frame is correct, could loop on that, mixed teams is the same
                # For some mixed doubles, the first frame is a set up, then they skip one, then they have numbering starting from 1
                # Some mixed doubles(CU_WMDCC2016P) just start off on throw 7 with three stones
                # Basically want to count on the frames
                # Could have different modes depending on if there is a frame in the (1, 0) slot
                
                # If there is a frame in the (1, 0) slot, then frame = ((x) + 6 * y) + 1
                # If there is not a frame in (1, 0), then frame = ((x) + 6 * y) - 1

                for i in range(row['num_positions'] + 1):
                    # Filter to only stones in the frame
                    stones_df_end_frame = stones_df_end.query('frame_num == @i')

                    # Make a position entry
                    # The frame_offset is used to account for the frame added for mens, womens, and one tournament
                    position_data = (i, end_id)
                    position_cmd = """
                    INSERT OR IGNORE INTO Position (frame_num, end_id)
                                            VALUES (?, ?)"""
                    db.execute_command(position_cmd, position_data)
                    position_id = db.cursor.lastrowid

                    if len(stones_df_end_frame) > 0:
                        # Make a stone entry for each stone, referencing the current position
                        stone_cmd = """
                        INSERT OR IGNORE INTO Stone (colour, x, y, size, position_id)
                                            VALUES (?, ?, ?, ?, ?)"""
                        for _, stone_row in stones_df_end_frame.iterrows():
                            stone_data = (stone_row['stone_colour'], stone_row['x'] * pixel_to_m, stone_row['y'] * pixel_to_m, stone_row['stone_size'], position_id)
                            db.execute_command(stone_cmd, stone_data, commit=False)
                    db.commit()

                # for i, (y, x) in enumerate(frames):
                #     # Filter to only stones in the frame
                #     stones_df_end_frame = stones_df_end.query('(frame_y == @y) & (frame_x == @x)')

                #     # Make a position entry
                #     # The frame_offset is used to account for the frame added for mens, womens, and one tournament
                #     position_data = (i + frame_offset, end_id)
                #     position_cmd = """
                #     INSERT OR IGNORE INTO Position (frame_num, end_id)
                #                             VALUES (?, ?)"""
                #     db.execute_command(position_cmd, position_data)
                #     position_id = db.cursor.lastrowid

                #     # Make a stone entry for each stone, referencing the current position
                #     stone_cmd = """
                #     INSERT OR IGNORE INTO Stone (colour, x, y, size, position_id)
                #                         VALUES (?, ?, ?, ?, ?)"""
                #     for _, stone_row in stones_df_end_frame.iterrows():
                #         stone_data = (stone_row['stone_colour'], stone_row['x'] * pixel_to_m, stone_row['y'] * pixel_to_m, stone_row['stone_size'], position_id)
                #         db.execute_command(stone_cmd, stone_data, commit=False)
                #     db.commit()
                        
        # Note that the tournament has been parsed
        with open(parsed_tournaments_path, 'a') as fh:
            fh.write(tournament + '\n')
                                
                

                
def get_stone_positions(filename: str, end_num: int) -> tuple[pd.DataFrame, str, bool]:
    """Take a full path to a file and return a dataframe with the stone positions.

    Args:
        filename (str): Full path to image to read.

    Returns:
        pd.DataFrame: Dataframe with columns [colour, x, y] containing all of the full-size stones.
    """

    logger.info(f"Parsing the images")

    # The actual stone detection is not 100%, there are some complicated things going on here.
    # Weaknesses include finding stones that are slightly overlapping with locations of previous stones, many of these are ignored
    # - This can be adjusted by the continuation thresholds

    lower_red = np.array([0, 250, 250])
    upper_red = np.array([0, 255, 255])
    lower_yellow = np.array([28, 200, 200])
    upper_yellow = np.array([32, 255, 255])
    lower_blue = np.array([120, 225, 225])
    upper_blue = np.array([130, 255, 255])
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([1, 1, 1])

    # Convert the image into a cv2 BGR object
    frame = cv2.imread(filename)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create the masks
    red_mask = cv2.inRange(hsv, lower_red, upper_red)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    yellow_mask = cv2.bitwise_or(yellow_mask, blue_mask)
    black_mask = cv2.inRange(hsv, lower_black, upper_black)

    stones = []
    hammer = ''

    for stone_colour, colour_mask in {'red': red_mask, 'yellow': yellow_mask}.items():
        # Find contours and bounding boxes
        contours, _ = cv2.findContours(colour_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        con_bound = [cv2.boundingRect(con) for con in contours]

        # -- Find the absolute locations of all of the stones --

        # Check for the hammer, checks the first cell
        if stone_colour == 'red':

            # First frame
            small_stones = [con for con in contours if (len(con) < 40) 
                                                   and (len(con) > 15)
                                                   and (con[0][0][1] > 1200)
                                                   and (con[0][0][1] < 2650)
                                                   and (con[0][0][0] > 280)
                                                   and (con[0][0][0] < 920)]

            # Count the number of 'small' stones to determine the hammer
            stone_num = len(small_stones)
            if any(stone_num == x for x in [5, 8]):
                hammer = 'red'
            elif any(stone_num == x for x in [4, 7]):
                hammer = 'yellow'
            else:
                hammer = 'incon'
            
            # Use the location of the small stones to identify the direction of play
            try:
                dir_of_play_down: bool = small_stones[0][0][0][1] < 1925
            except:
                logger.error(f"Can't find the direction of play by counting the small stones.")
                logger.error(end_num)
                logger.error(hammer)
                logger.error(small_stones)

                quit()

        # For enumerated contours, if contour centre is inside only one bounding box, then add it to good list
        valid_con = []
        for i, con in enumerate(contours):
            # If bounding box is too small or large, skip, if the length of the contour is too small or large, skip
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
                valid_con.append(con)
            
        # Find the centres of the rocks
        # This is an averaging of the bounding box and mean
        for con in valid_con:
            con_unwrap = [x[0] for x in con]
            centroid_x = round(sum([x[0] for x in con_unwrap])/len(con_unwrap))
            centroid_y = round(sum([x[1] for x in con_unwrap])/len(con_unwrap)) 

            x, y, w, h = cv2.boundingRect(con)

            avg_x = round((centroid_x + x + w/2)/2)
            avg_y = round((centroid_y + y + h/2)/2)
        
            stones.append((stone_colour, end_num, avg_x, avg_y, len(con_unwrap)))
    
    # -- Find the houses --
    # Find the outlines of the frames, they slightly vary in size
    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter to only the frame bounding boxes
    bounding = [bound for bound in [cv2.boundingRect(con) for con in contours] if (bound[2] > 500) and (bound[3] > 900)]

    # Here I could number each of the frames for the stones

    # If there is no (1, 0), then (0, 0) is 0, (2, 0) is 1, ...
    # If there is a (1, 0), then (0, 0) is 1 and the frame insertion part of the main loop still applies

    # Construct a list of the houses/frames, noting the index of the frame and location of the house
    houses = []
    for bound in bounding:
        orig_x = bound[0] + (bound[2] - 3)/2
        if dir_of_play_down:
            orig_y = bound[1] + 907 # 907.5 - Bottom House
        else:
            orig_y = bound[1] + 328 # 328.5 - Top House
        
        # Find the location index
        ind_x = (bound[0] - 288) // 730
        # ind_y = (bound[1] - 1355) // 1438
        ind_y = (bound[1] - 1330) // 1410

        # logger.info(f"x: {ind_x}, y: {ind_y}")
        houses.append((end_num, ind_x, ind_y, orig_x, orig_y))
    
    # This is a test if there is a (1, 0) in the houses
    start_normal: bool = True if any(((house[1] == 1) and (house[2] == 0)) for house in houses) else False

    def frame_num_calc(x, y, normal_count: bool) -> int:
        """Return the frame number"""
        if normal_count:
            return int(x + (6 * y) + 1)
        else:
            return int(x + (6 * y) - 1)

    # Here we take the tuples in, and just add a field to them
    houses_new = []
    for house in houses:
        calc_frame_num = frame_num_calc(house[1], house[2], start_normal)
        if calc_frame_num == -1: # Guard for the weird counting, give frame num of 0 instead of -1
            houses_new.append((*house, 0))
        else:
            houses_new.append((*house, calc_frame_num))
    houses = houses_new
    # Highest index of position on page
    num_positions = max([house[5] for house in houses])
    logger.info(f"Max index of positions: {num_positions}, normal? {start_normal}")
    
    # -- Find positions of stones in houses --
    # For each frame, find position of each stone relative to centre of house
    parsed_stones = []
    for bound, (_, ind_x, ind_y, orig_x, orig_y, calc_frame_num) in zip(bounding, houses):
        for stone in stones:
            # For each stone inside the frame
            if (stone[2] > bound[0]) and (stone[2] < (bound[0] + bound[2])) and (stone[3] > (bound[1] + 42)) and (stone[3] < (bound[1] + bound[3] - 42)):
                if dir_of_play_down:
                    parsed_stones.append((end_num, calc_frame_num, ind_x, ind_y, stone[0], (stone[2] - orig_x), -(stone[3] - orig_y), stone[4]))
                else:
                    parsed_stones.append((end_num, calc_frame_num, ind_x, ind_y, stone[0], -(stone[2] - orig_x), (stone[3] - orig_y), stone[4]))
        
    # -- Return stone data --
    logger.info(f"Completed stone parsing page {filename.split('/')[-1]}")
    return pd.DataFrame(parsed_stones, columns=['end', 'frame_num', 'frame_x', 'frame_y', 'stone_colour', 'x', 'y', 'stone_size']), hammer, dir_of_play_down, num_positions

    
def parse_text(filename, temp_dir: str) -> tuple[str, tuple, pd.DataFrame, pd.DataFrame]:

    logger.info(f"Parsing the xml")

    # Zones and buffers
    shot_buffers = {'left': 20, 'right': 115, 'up': 240, 'down': 13}
    date_box = {'left': 0, 'right': 360, 'top': 80, 'bottom': 210}
    headers_box = {'top': 0, 'bottom': 80}
    sheet_pattern = re.compile(f'.*Sheet ([a-zA-Z0-9])')

    # Lists to be created with appends
    page_info = []
    shot_info = []

    # Get the root of the xml file
    root = et.parse(join(temp_dir, filename)).getroot()

    # Initialize the per-document fields, some of them might not parse correctly
    start_time = ''
    date = None
    sheet = ''
    title = ''
    header_parse = []
    rating_mapping = {'100%': 4, '75%': 3, '50%': 2, '25%': 1, '0%': 0}

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
                # TODO: There is also a 'Shoot-out' option which is just a closest to the centre
                if ('Prepositioned Stones' in item.text):
                    pass
                if (': ' in item.text):
                    # These are all of the shots
                    # Find the shot index
                    ind_x = ((item_left - 50) // 130) + 1
                    ind_y = (item_top - 480) // 250
                    shot_info.append([page_num, ind_x, ind_y])
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
                                # Convert to integer, 4 is best, -1 indicates error reading
                                # throw_rating = element.text
                                if '%' in element.text:
                                    throw_rating = rating_mapping.get(element.text.strip(), -1)
                                elif (len(element.text) == 1):
                                    try:
                                        throw_rating = int(element.text)
                                    except:
                                        throw_rating = -1
                                        
                                    
                            else:
                                # Else it is the throw turn direction
                                throw_turn = element.text
                                
                    shot_info[-1].extend([player, throw_num, throw_type, throw_rating, throw_turn])
                
                # -- Parse for End Score --
                # This is for the more common string case
                # TODO: Apparently the resulting number can be W/L, should just take score of previous end
                if ('=' in item.text) and (abs(item_top - 225) < 5):
                    score_tuple = (int(item.attrib['top']), int(item.attrib['left']), item.text.split()[-1].strip(' ='))
                    # If either of the parsed scores are W/L then it will go with the 99-99 option which indicates error parsing
                    # TODO: In the future with some testing this could take the previous scores with team_1_score and team_2_score
                    if not any(x in score_tuple[2] for x in ['W', 'L']):
                        banner_parse.append((score_tuple[0], score_tuple[1], int(score_tuple[2])))
                
                # Less common split case
                # TODO: Apparently the added number can also take the value of X
                if (abs(item_top - 225 < 5)) and (len(item.text.strip()) < 3) and (len(item.text.strip()) > 0) and all(x not in item.text for x in ['+', '-', 'X', '/']):
                    if item.text.strip() == 'X':
                        banner_parse.append((int(item.attrib['top']), int(item.attrib['left']), 0))
                    else:
                        banner_parse.append((int(item.attrib['top']), int(item.attrib['left']), int(item.text.strip())))
                
                # -- Parse for Title Info --
                # Only check first page
                if int(page.attrib['number']) == 1:
                    # Check for the start time
                    if 'Start Time' in item.text:
                        start_hour, start_minutes = item.text.strip().split()[-1].split(':')
                        start_time = datetime.timedelta(hours=int(start_hour), minutes=int(start_minutes))

                    # Check for the date
                    if (len(item.text.strip().split()) == 4) and (item_top > date_box['top']) and (item_top < date_box['bottom']) and (item_left < date_box['right']):
                        date = dateparser.parse(item.text.strip())
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
    
    page_df = pd.DataFrame(page_info, columns=['end_number', 'team_1_score', 'team_2_score'])
    shot_df = pd.DataFrame(shot_info, columns=['end_number', 'ind_x', 'ind_y', 'player', 'throw_num', 'throw_type', 'throw_rating', 'throw_turn'])

    # Parse the final score of the match
    team_1_final_score = page_df['team_1_score'].max()
    team_2_final_score = page_df['team_2_score'].max()

    if isinstance(date, datetime.datetime) and isinstance(start_time, datetime.timedelta):
        start_time = date + start_time
    elif isinstance(date):
        logger.warning(f"No start time found.")
        start_time = date

    tourney_info = title
    doc_info = (start_time, sheet, team_1_final_score, team_2_final_score)

    return tourney_info, doc_info, page_df, shot_df
                    

def pdf_to_xml(filename: str, pdf_dir: str, temp_dir: str):
    """Convert given pdf to xml, requires full filepath"""

    cmd = f"pdftohtml -xml {join(pdf_dir, filename).strip()} {join(temp_dir, filename[:-4])}"
    logger.info(f"Creating xml from pdf: {cmd}")
    system(cmd)


def pdf_to_xml_all():
    """Cycle through pdfs and create xmls using pdftohtml with the xml option."""

    if not isdir(join(base_directory, 'shots_xml')):
        mkdir(join(base_directory, 'shots_xml'))

    pdfs = [x.strip().replace(r"'", r"\'") for x in listdir(join(base_directory, 'shots'))]
    for pdf in tqdm(pdfs):
        cmd = f"pdftohtml -xml {join(base_directory, 'shots', pdf).strip()} {join(base_directory, 'shots_xml', pdf[:-4])}"
        logger.info(f"Executing command: {cmd}")
        system(cmd)


def remove_spans(filename, temp_dir: str):
    """There are some weird spans introduced in the parsing but this will remove them in place.

    Args:
        filename (str): The full filename of the file to remove the spans from
    """
    logger.info(f"Removing spans from xml: {filename}")
    with open(join(temp_dir, filename), 'r', encoding='ISO-8859-1') as fh:
        xml_text = fh.read()
        xml_text = re.sub(r'</span>.*">', ' ', xml_text)

    with open(join(temp_dir, filename), "w", encoding='ISO-8859-1') as fh:
        fh.write(xml_text)


def pdf_to_img(filename: str, pdf_dir: str, temp_dir: str, zoom: int=8):
    """Take a file name and convert all pages to images, saved in the sample_images directory

    Args:
        filename (str): filename of pdf to convert to images
    """

    logger.info(f"Creating images from pdf pages")
    doc = fitz.open(join(pdf_dir, filename))
    # Parse the pdf into images
    for i, page in enumerate(doc.pages()):
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        pix.save(join(temp_dir, f"page_{i + 1}.png"))
        logger.info(f"Completed page/image {i + 1}")


if __name__ == '__main__':
    main()
