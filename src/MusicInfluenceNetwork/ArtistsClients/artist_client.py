import requests
import json
from abc import ABC, abstractmethod
from typing import Optional

from MusicInfluenceNetwork.Models import Artist
    

class ArtistClient(ABC):
    
    def __init__(self, user_agent: str):
        self.user_agent = user_agent
    
    @abstractmethod
    def get_artists_by_genre(self, genre: str, num_requests: int = 1000, limit: int = 100, is_sleep: bool = False, offset: int = 0) -> list[Artist]:
        pass

    def stream_artists_genre(self, genre, num_requests=1000):

        data = self.get_artists_by_genre(genre, num_requests)
        
        for artist in data:
            yield artist