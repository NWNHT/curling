
-- Event_name, date, location(maybe), start and end date
CREATE TABLE Event (
    event_id INTEGER NOT NULL PRIMARY KEY,
    name TEXT,
    abbrev TEXT UNIQUE,
    start_date TEXT,
    end_date TEXT,
    location TEXT
);

-- Include time control? Include a final score?
CREATE TABLE Match (
    match_id INTEGER NOT NULL PRIMARY KEY,
    start_time TEXT,
    type TEXT,
    sheet TEXT,
    team_1 TEXT,
    team_1_final_score INTEGER,
    team_2 TEXT,
    team_2_final_score INTEGER,

    event_id INTEGER,
    FOREIGN KEY(event_id) REFERENCES Event(event_id)
);

CREATE TABLE End (
    end_id INTEGER NOT NULL PRIMARY KEY,
    num INTEGER NOT NULL,
    hammer_colour TEXT,
    direction TEXT,

    -- Red
    team_1_final_score INTEGER,
    -- Yellow
    team_2_final_score INTEGER,
    
    match_id INTEGER,
    FOREIGN KEY(match_id) REFERENCES Match(match_id)
);

CREATE TABLE Player (
    player_id INTEGER NOT NULL PRIMARY KEY,
    name TEXT,
    sex TEXT,
    team TEXT,

    UNIQUE (name, sex, team)
);

CREATE TABLE Position (
    position_id INTEGER NOT NULL PRIMARY KEY,
    frame_num INTEGER NOT NULL,

    end_id INTEGER,
    FOREIGN KEY(end_id) REFERENCES End(end_id)

);

CREATE TABLE Stone (
    stone_id INTEGER NOT NULL PRIMARY KEY,
    colour TEXT,
    x REAL,
    y REAL,

    position_id INTEGER,
    FOREIGN KEY(position_id) REFERENCES Position(position_id)
);

CREATE TABLE Throw (
    throw_id INTEGER NOT NULL PRIMARY KEY,
    throw_num INTEGER NOT NULL,
    colour TEXT,
    rating INTEGER,
    
    player_id INTEGER,
    end_id INTEGER,

    FOREIGN KEY(player_id)  REFERENCES Player(player_id),
    FOREIGN KEY(end_id)     REFERENCES End(end_id)
);
