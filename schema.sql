
CREATE TABLE ArtistGroup (
    group_id INTEGER PRIMARY KEY AUTO_INCREMENT
);

CREATE TABLE Artist (
    artist_id INTEGER PRIMARY KEY AUTO_INCREMENT,
    mbid VARCHAR(64) UNIQUE,
    artist_name VARCHAR(255) UNIQUE,
    career_start_year INTEGER,
    career_end_year INTEGER,
    is_weak_node BOOLEAN,
    is_explored BOOLEAN,
    group_id INTEGER,
    FOREIGN KEY (group_id) REFERENCES ArtistGroup(group_id)
);

CREATE TABLE Genre (
    genre_id INTEGER PRIMARY KEY AUTO_INCREMENT,
    genre_name VARCHAR(256) NOT NULL
);

CREATE TABLE ArtistGenre (
    artist_id INTEGER,
    genre_id INTEGER,
    FOREIGN KEY (artist_id) REFERENCES Artist(artist_id),
    FOREIGN KEY (genre_id) REFERENCES Genre(genre_id),
    PRIMARY KEY (artist_id, genre_id)
);

CREATE TABLE SimilarArtist (
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    artist_id INTEGER NOT NULL, -- Root artist
    similar_artist_id INTEGER NOT NULL,
    FOREIGN KEY (artist_id) REFERENCES Artist(artist_id),
    FOREIGN KEY (similar_artist_id) REFERENCES Artist(artist_id),
    lastFM_similarity FLOAT NOT NULL,
    spotify_similarity FLOAT NOT NULL
);


-- Delete duplicates in Artist:
DELETE FROM ArtistGenre
WHERE artist_id NOT IN (
    SELECT MIN(artist_id) 
    FROM Artist 
    GROUP BY artist_name
);
WITH cte AS (
    SELECT MIN(artist_id) AS artist_id
    FROM Artist
    GROUP BY artist_name
)
DELETE FROM Artist
WHERE artist_id NOT IN (SELECT artist_id FROM cte);