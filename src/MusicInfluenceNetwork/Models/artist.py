from dataclasses import dataclass

@dataclass
class Artist:
    name: str
    mbid: str
    url: str
    begin: int


@dataclass
class SimilarArtist(Artist):
    match: float

