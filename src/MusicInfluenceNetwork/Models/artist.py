from dataclasses import dataclass

@dataclass
class Artist:
    name: str
    mbid: str
    url: str


@dataclass
class SimilarArtist(Artist):
    match: float

