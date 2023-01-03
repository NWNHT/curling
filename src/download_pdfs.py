
from os import mkdir
from os.path import isdir, isfile, join
from pathlib import Path
import re
from time import sleep

import requests as req
from tqdm import tqdm

# The base url for the data
base_url = "http://odf2.worldcurling.co"

base_directory = str(Path(__file__).parent.parent.resolve())

def create_shots_file():
    """Read all tournaments in the base url and create a file containing all shot_by_shot files
    """

    # Get the list of items in /data/ and filter for just directories
    base = req.get(base_url + "/data/").text
    base = '\n'.join([x for x in base.split('<br>') if 'dir' in x])
    # regex for the paths
    data_pattern = re.compile(r'.*HREF="(/data/.*)".*')
    pot_tournies = data_pattern.findall(base)

    # Loop through all of the sessions for all 'teams' for lall tournaments and 
    #   find files with names containing 'Shot_by_Shot', save to list all_shots
    all_shots = []
    for pot_tourney in tqdm(pot_tournies):
        
        # Request the page of the potential tournament, filter and format a bit
        tourney_content = req.get(base_url + pot_tourney).text
        tourney_content = '\n'.join([x for x in tourney_content.split('<br>') if 'dir' in x])
        # regex for the specific paths, the 'teams'
        team_pattern = re.compile(r'.*HREF="(/data/.*(?:Men\'s_Teams|Women\'s_Teams|Mixed_Teams|Mixed_Doubles)/)".*')
        teams = team_pattern.findall(tourney_content)

        for team in teams:

            # Request the page of each team, filter out just the directories, these should be the 'sessions'
            team_content = req.get(base_url + team).text
            team_content = '\n'.join([x for x in team_content.split('<br>') if 'dir' in x])
            # We just need to regex for the paths here, no filtering is done as we want all of the directories
            session_pattern = re.compile(r'.*HREF="(/data/.*)".*')
            sessions = session_pattern.findall(team_content)

            for session in sessions:

                # Request the page of each session
                session_content = req.get(base_url + session).text
                session_content = '\n'.join([x for x in session_content.split('<br>')])
                # Find all paths containing 'shot by shot'
                shot_pattern = re.compile(r'.*HREF="(/data/.*Shot_by_Shot.*)".*')
                shots = shot_pattern.findall(session_content)

                # Append all shotbyshots to the list of all of them
                all_shots.extend(shots)
                print(f"Found {len(shots)}, Total: {len(all_shots)}")

                # Out of the kindness of my heart
                sleep(0.5)

    # Write the list to a file
    with open(join(base_directory, 'all_shots.txt'), 'w') as fh:
        fh.write('\n'.join(all_shots))


def download_shots():

    # Read all_shots file for all shot by shot files
    try:
        with open(join(base_directory, 'all_shots.txt'), 'r') as fh:
            all_shots = [x.strip() for x in fh.readlines()]
    except Exception as e:
        print(f"Cannot open shots file with exception: {e}")
    
    # Make data directory if does not exist
    if not isdir(join(base_directory, 'shots')):
        mkdir(join(base_directory, 'shots'))
    
    # For each filename, check if already downloaded and download if not
    for shot in tqdm(all_shots):
        fc = shot.strip().split('/')
        filename = f"{fc[2]}-{fc[3]}-{fc[4]}-{fc[5].split('_')[-1]}"

        # Skip if the file has already been downloaded
        if isfile(join(base_directory, 'shots', filename)):
            print(f"File {filename} already present.")
            continue
        print(f"Downloading {filename}")

        try:
            # Download the file and save with filename
            summary = req.get(base_url + shot)

            # Write the file
            with open(join(base_directory, 'shots', filename), 'wb') as fh:
                fh.write(summary.content)
        except Exception as e:
            print(f"Error downloading file {filename}, with exception : {e}")
        
        # I am no sadist
        sleep(2)
        

if __name__ == '__main__':
    # Create list of shot_by_shot files to download
    create_shots_file()

    # Download all of the files in the list
    download_shots()
