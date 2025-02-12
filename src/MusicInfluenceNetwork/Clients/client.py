import requests
import json
from abc import ABC, abstractmethod
from typing import Optional

from MusicInfluenceNetwork.Models import SimilarArtist
    

class Client(ABC):
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    @abstractmethod
    def get_similar_artists(self, artist: Optional[str] = None, mbid: Optional[str] = None) -> list[SimilarArtist]:
        pass
    



