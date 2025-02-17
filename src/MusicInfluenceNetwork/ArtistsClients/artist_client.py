import requests
import json
from abc import ABC, abstractmethod
from typing import Optional
    

class ArtistClient(ABC):
    
    @abstractmethod
    def artists_genre(self, genre: str, num_requests: int = 1000, limit: int = 100, is_sleep: bool = False, offset: int = 0) -> list[dict]:
        pass

    def stream_artists_genre(self, genre, num_requests=1000, limit=100, is_sleep=False, offset=0):

        data = self.artists_genre(genre, num_requests, limit, is_sleep, offset)
        
        for artist in data:
            yield {
                "name": artist.get("name", {}),
                "id": artist.get("id", {}),
                "birth_year": artist.get("life-span", {}).get("begin"),
            }