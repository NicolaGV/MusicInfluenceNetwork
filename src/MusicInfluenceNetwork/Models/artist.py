from dataclasses import dataclass
from typing import Optional

    
@dataclass
class Artist:
    name: str
    artist_id: Optional[int] = None
    mbid: Optional[str] = None
    career_start_year: Optional[int] = None
    career_end_year: Optional[int] = None
    
    def __hash__(self):
        return hash(self.name)
            
    def __eq__(self, other):
        return self.name == other.name
    
    @staticmethod
    def from_musicbrainz(mb_json: dict) -> 'Artist':
        begin_year = mb_json["life-span"].get("begin", None)
        end_year = mb_json["life-span"].get("end", None)
        
        if begin_year:
            begin_year = int(begin_year[:4]) # Only the year
        if end_year:
            end_year = int(end_year[:4]) # Only the year
        
        return Artist(mb_json["name"], mb_json["id"], begin_year, end_year)
    
    @staticmethod
    def from_db_row(row: tuple) -> 'Artist':
        return Artist(name=row[2], artist_id=row[0], mbid=row[1], career_start_year=row[3], career_end_year=row[4])


@dataclass
class SimilarArtist(Artist):
    last_fm_match: Optional[float] = None
    spotify_match: Optional[float] = 0
    
    def __hash__(self):
        return hash(self.name)
            
    def __eq__(self, other):
        return self.name == other.name

