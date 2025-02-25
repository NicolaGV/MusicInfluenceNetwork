


CREATE TABLE Artist (
    artist_id INTEGER PRIMARY KEY AUTO_INCREMENT,
    mbid INTEGER UNIQUE,
    artist_name TEXT NOT NULL,
    career_start_year INTEGER,
    career_end_year INTEGER
);

CREATE TABLE Genre (
    genre_id INTEGER PRIMARY KEY,
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
    artist_id INTEGER NOT NULL,
    similar_artist_id INTEGER NOT NULL,
    FOREIGN KEY (artist_id) REFERENCES Artists(id),
    FOREIGN KEY (similar_artist_id) REFERENCES Artists(id),
    lastFM_similary FLOAT NOT NULL,
    spotify_similarity FLOAT NOT NULL
);