import requests
import json
from abc import ABC, abstractmethod
from typing import Optional
    

class ArtistClient(ABC):
    
    def __init__(self, user_agent: str):
        self.user_agent = user_agent
    
    @abstractmethod
    def artists_genre(self, genre: str, num_requests: int = 1000) -> list[dict]:
        pass

    def stream_artists_genre(self, genre, num_requests=1000):

        data = self.artists_genre(genre, num_requests)
        
        for artist in data:
            yield {
                "name": artist.get("name", {}),
                "id": artist.get("id", {}),
                "birth_year": artist.get("life-span", {}).get("begin"),
            }