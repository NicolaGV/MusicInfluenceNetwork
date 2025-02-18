from dataclasses import dataclass
from typing import Optional

    
@dataclass
class Artist:
    name: str
    mbid: str
    url: str
    begin: Optional[int] = None


@dataclass
class SimilarArtist(Artist):
    match: Optional[float] = None

