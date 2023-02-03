from itertools import product, permutations
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotnine as gg
from tqdm import tqdm
from typing import Tuple

from CurlingDB import CurlingDB
from utils.PlotnineElements import PlotnineElements as pe, blank


def find_pos_id(db: CurlingDB, event: str, team_1: str, team_2: str, end: int):
    db.execute_command(find_pos_id_query, (event, team_1, team_2, end))
    return pd.DataFrame(db.cursor.fetchall(), columns=['frame_num', 'position_id'])
    

def position_stones(db: CurlingDB, position_id: int) -> Tuple[list, list, Tuple[int, int, int]]:

    # Execute query on db and make dataframe
    db.execute_command(position_stones_query, (position_id,))
    example_stones = pd.DataFrame(db.cursor.fetchall(), columns=['s_id', 'stone_team', 'x', 'y', 'throw_team'])

    # Create the target lists for the two teams
    thrower_targets = example_stones.query('stone_team == throw_team')[['x', 'y']].values
    non_thrower_targets = example_stones.query('stone_team != throw_team')[['x', 'y']].values

    # Stone counts for finding similar positions
    stone_counts = (len(thrower_targets), len(non_thrower_targets), len(thrower_targets) + len(non_thrower_targets))
    
    return thrower_targets, non_thrower_targets, stone_counts


def find_similar_positions(db: CurlingDB, stone_counts: tuple) -> pd.DataFrame:

    # Fetch positions and turn into dataframe
    db.execute_command(find_similar_positions_query, stone_counts)
    positions = pd.DataFrame(db.cursor.fetchall(), columns=['pos', 'next_pos', 'frame_num', 'stone_id', 'stone_team', 'x', 'y', 'size', 'type', 'throw_team', 'match_win', 'end_win']).dropna()

    return positions


def add_scores(positions: pd.DataFrame, thrower_targets, non_thrower_targets, include_inverse: bool=False):
    
    # Initialization
    pos_scores = [(1, 1e3)] * positions.pos.unique().shape[0]
    positions_int = positions.copy(deep=True)
    parse_count = 2 if include_inverse else 1
    
    for _ in range(parse_count):
        for i, (pos, group) in enumerate(positions_int.groupby('pos')):

            # Collect stone positions for each team
            throwing_team = group.iloc[0]['throw_team']
            pos_thrower_stones = group.query('stone_team == @throwing_team')[['x', 'y']].values
            pos_non_thrower_stones = group.query('stone_team != @throwing_team')[['x', 'y']].values

            pos_score = 0
            max_stone_score = 0
            for points, stones in [(non_thrower_targets, pos_non_thrower_stones), (thrower_targets, pos_thrower_stones)]:
                N = len(points)
                if N == 0: 
                    # No stones, no loss
                    continue
                elif N == 1:
                    # Handle the single stone case, single permutation
                    pos_score += (((points - stones)**2).sum())**(1/2)
                    max_stone_score = (((points - stones)**2).sum())**(1/2) if (((points - stones)**2).sum())**(1/2) > max_stone_score else max_stone_score
                    continue

                # Create the matrix of distance between each of the points
                indices = list(zip(*[x for x in product(range(2*N), range(2*N)) if (sum(x) % 2)]))
                grid = np.subtract.outer(stones.ravel(), points.ravel())
                grid[indices[0], indices[1]] = 0  # Zero out the x-y operations
                grid_sum = (grid**2).reshape(N, 2, N, 2).sum(axis=1).sum(axis=-1)**(1/2) # Element-wise square, sum along axes to reduce, and then element-wise sqrt

                # Find the best permutation of the elements
                # This will be used to determine if the position is a match, match-ness described by the distance
                # Alternatively, you could just loop over stones in the actual position and check if there are other stones within their radius but
                #   I am not totally sure this would work as it would depend on the parsing order of the stones.
                # - Another idea to implement is a partial match where only the house or only a certain slice of the ice is considered
                #   - This would have to be implemented earlier and likely in SQL if I don't want to pull all the data
                
                # Create list of permutations
                perms = list(permutations(range(N), N))
                # Get the 'loss' for each permutation of stone-stone matching and add to running sum
                losses = grid_sum[range(N), np.array(perms)].sum(-1)
                pos_score += losses.min() / group.shape[0]
                # Get the stone order for the closest match and then the maximum stone score(farthest stone from target)
                stone_order = perms[losses.argmin()] 
                max_stone_score = grid_sum[range(N), stone_order].max() if grid_sum[range(N), stone_order].max() > max_stone_score else max_stone_score

            # pos_scores.append((pos, pos_score, max_stone_score))
            if pos_score < pos_scores[i][1]:
                pos_scores[i] = (pos, pos_score, max_stone_score)
        
        # If the parse_count is two then the second round will be inverted
        # Flip all of the positions horizontally to check the symmetric positions
        positions_int['x'] = positions_int['x'] * -1
        
        
    scores_df = pd.DataFrame(pos_scores, columns=['pos', 'score', 'max_stone_score'])
    return scores_df


def next_position_stones(db, positions, score_threshold, total_count):

    # Extract the position_ids
    next_positions = positions.query('score < @score_threshold').next_pos.unique()
    # next_position_stones_query += ','.join('?' * len(next_positions)) + ')'

    # Get the stones and create dataframe
    db.execute_command(next_position_stones_query + ','.join('?' * len(next_positions)) + ')', tuple(next_positions))
    next_stones = pd.DataFrame(db.cursor.fetchall(), columns=['pos', 'frame_num', 'stone_team', 'colour', 'x', 'y', 'size', 'type', 'rating', 'throw_team', 'match_win', 'end_win']).dropna()

    # Filter to valid positions by removing any with too many stones
    valid_positions = next_stones.groupby('pos').count().query('stone_team < @total_count + 2').index.values
    next_position_stones = next_stones.query('pos in @valid_positions')

    # Add a label column:w
    next_position_stones['plot_label'] = next_position_stones.apply(lambda x: f"{x.frame_num}:{x.type}:{x.rating}:{'Won' if x.throw_team == x.end_win else 'Lost'}", axis=1)

    next_position_stones = next_position_stones.sort_values(by=['frame_num', 'type', 'rating'])

    return next_position_stones
    

def similar_positions(db: CurlingDB, position_id: int, score_threshold, include_inverse: bool=False):

    thrower_targets, non_thrower_targets, stone_counts = position_stones(db, position_id)
    print(f"Finding positions with {stone_counts[0]} thrower stones and {stone_counts[1]} non-thrower stones.")

    positions = find_similar_positions(db, 
                                       stone_counts)
    print(f"Found {len(positions.pos.unique())} positions with the same stone count.")
    positions = positions.merge(add_scores(positions, 
                                           thrower_targets, 
                                           non_thrower_targets,
                                           include_inverse=include_inverse))
    print(f"Score calculated for all positions.")

    next_stones = next_position_stones(db, positions, score_threshold, stone_counts[2])
    print(f"Found {len(next_stones.pos.unique())} positions with {len(next_stones)} stones.")

    return positions, thrower_targets, non_thrower_targets, next_stones


find_pos_id_query = """
SELECT
    p.frame_num,
    p.position_id
FROM
    Position p
JOIN
    End e
ON
    p.end_id = e.end_id
JOIN
    Match m
ON
    e.match_id = m.match_id
JOIN
    Event e2
ON
    m.event_id = e2.event_id
WHERE
    e2.abbrev = ?
AND
    m.team_1 = ?
AND
    m.team_2 = ?
AND
    e.num = ?
"""


position_stones_query = """
SELECT
    s.stone_id,
    CASE
        WHEN
            s.colour = 'red'
        THEN
            'team1'
        WHEN
            s.colour = 'yellow'
        THEN
            'team2'
        ELSE
            'unknown'
        END,
    s.x,
    s.y,
    CASE
        WHEN
            t.colour = 'red'
        THEN
            'team2'
        WHEN
            t.colour = 'yellow'
        THEN
            'team1'
        ELSE
            'unknown'
        END
FROM 
    Stone s
JOIN
    Position p
ON
    s.position_id = p.position_id
JOIN
    End e
ON
    p.end_id = e.end_id
JOIN
    Throw t
ON
    e.end_id = t.end_id AND p.frame_num = t.throw_num
JOIN
    Match m
ON
    e.match_id = m.match_id
JOIN
    Event e2
ON
    m.event_id = e2.event_id
WHERE p.position_id = ?
"""


find_similar_positions_query = """
SELECT
    p.position_id,
    p.next_pos,
    p.frame_num,
    s.stone_id,
    CASE
        WHEN
            s.colour = 'red'
        THEN
            'team1'
        WHEN
            s.colour = 'yellow'
        THEN
            'team2'
        ELSE
            'unknown'
        END,
    s.x,
    s.y,
    s.size,
    t.type,
    CASE
        WHEN
            t.colour = 'red'
        THEN
            'team2'
        WHEN
            t.colour = 'yellow'
        THEN
            'team1'
        ELSE
            'unknown'
        END,
    CASE
        WHEN
            m.team_1_final_score > m.team_2_final_score
        THEN
            'team1'
        WHEN
            m.team_1_final_score < m.team_2_final_score
        THEN
            'team2'
        ELSE
            'draw'
        END,
    CASE 
		WHEN
			(e.team_1_final_score - COALESCE(LAG(e.team_1_final_score, 1)  OVER (PARTITION BY m.match_id ORDER BY e.end_id), 0) - 
			e.team_2_final_score - COALESCE(LAG(e.team_2_final_score, 1)  OVER (PARTITION BY m.match_id ORDER BY e.end_id), 0)) > 0
		THEN
			'team1'
		WHEN
			(e.team_1_final_score - COALESCE(LAG(e.team_1_final_score, 1)  OVER (PARTITION BY m.match_id ORDER BY e.end_id), 0) - 
			e.team_2_final_score - COALESCE(LAG(e.team_2_final_score, 1)  OVER (PARTITION BY m.match_id ORDER BY e.end_id), 0)) < 0
		THEN
			'team2'
		ELSE
			'blank'
		END
FROM 
    Stone s
JOIN
    (SELECT
        *,
        LEAD(position_id, 1) OVER (PARTITION BY end_id ORDER BY frame_num) AS next_pos
    FROM
        Position) p
ON
    s.position_id = p.position_id
JOIN
    End e
ON
    p.end_id = e.end_id
JOIN
    Throw t
ON
    e.end_id = t.end_id AND p.frame_num = t.throw_num
JOIN
    Match m
ON
    e.match_id = m.match_id
JOIN
    Event e2
ON
    m.event_id = e2.event_id
JOIN
    (SELECT
        p.position_id,
        e.end_id,
        p.frame_num AS frame_num,
        COUNT(*)
    FROM 
        Position p
    JOIN
        Stone s
    ON
        s.position_id = p.position_id
    JOIN
        End e
    ON
        p.end_id = e.end_id
    JOIN
        Throw t
    ON
        e.end_id = t.end_id AND p.frame_num = t.throw_num
    JOIN
        Match m
    ON
        e.match_id = m.match_id
    JOIN
        Event e2
    ON
        m.event_id = e2.event_id
    GROUP BY
        1
    HAVING
        COUNT(CASE WHEN s.colour != t.colour THEN 1 ELSE NULL END) = ? AND
        COUNT(CASE WHEN s.colour = t.colour THEN 1 ELSE NULL END) = ? AND
        COUNT(*) = ?
    ) sub
ON 
    p.frame_num = sub.frame_num AND e.end_id = sub.end_id
"""


next_position_stones_query = """
SELECT
    p.position_id,
    p.frame_num,
    CASE
        WHEN
            s.colour = 'red'
        THEN
            'team1'
        WHEN
            s.colour = 'yellow'
        THEN
            'team2'
        ELSE
            'unknown'
        END,
    CASE WHEN s.size < 47 THEN (CASE WHEN s.colour == 'red' THEN 'pink' WHEN s.colour == 'yellow' THEN 'green' END)
        ELSE s.colour END AS label,
    s.x,
    s.y,
    s.size,
    t.type,
    t.rating,
    CASE
        WHEN
            t.colour = 'red'
        THEN
            'team1'
        WHEN
            t.colour = 'yellow'
        THEN
            'team2'
        ELSE
            'unknown'
        END,
    CASE
        WHEN
            m.team_1_final_score > m.team_2_final_score
        THEN
            'team1'
        WHEN
            m.team_1_final_score < m.team_2_final_score
        THEN
            'team2'
        ELSE
            'draw'
        END,
    CASE 
		WHEN
			(e.team_1_final_score - COALESCE(LAG(e.team_1_final_score, 1)  OVER (PARTITION BY m.match_id ORDER BY e.end_id), 0) - 
			e.team_2_final_score - COALESCE(LAG(e.team_2_final_score, 1)  OVER (PARTITION BY m.match_id ORDER BY e.end_id), 0)) > 0
		THEN
			'team1'
		WHEN
			(e.team_1_final_score - COALESCE(LAG(e.team_1_final_score, 1)  OVER (PARTITION BY m.match_id ORDER BY e.end_id), 0) - 
			e.team_2_final_score - COALESCE(LAG(e.team_2_final_score, 1)  OVER (PARTITION BY m.match_id ORDER BY e.end_id), 0)) < 0
		THEN
			'team2'
		ELSE
			'blank'
		END
    
FROM 
    Stone s
JOIN
    Position p
ON
    s.position_id = p.position_id
JOIN
    End e
ON
    p.end_id = e.end_id
JOIN
    Throw t
ON
    e.end_id = t.end_id AND t.throw_num = p.frame_num
JOIN
    Match m
ON
    e.match_id = m.match_id
JOIN
    Event e2
ON
    m.event_id = e2.event_id
WHERE
    p.position_id IN ("""
