from dataclasses import dataclass
from typing import Optional

@dataclass
class SimilarArtist:
    artist_id: Optional[int] = None
    similar_artist_id: Optional[int] = None
    last_fm_match: Optional[float] = None
    spotify_match: Optional[float] = 0
    
    def __hash__(self):
        return hash(self.artist_id)
            
    def __eq__(self, other):
        return self.artist_id == other.artist_id

    @staticmethod
    def from_db_row(row):
        return SimilarArtist(artist_id=row[0], similar_artist_id=row[1], last_fm_match=row[2], spotify_match=row[3])