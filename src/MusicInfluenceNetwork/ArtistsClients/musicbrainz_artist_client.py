from ArtistsClients import ArtistClient

import requests
import time
import json

import os
from dotenv import load_dotenv

from pathlib import Path


class MusicbrainzArtistClient(ArtistClient):

    def __init__(self, base_url: str = "https://musicbrainz.org/ws/2/artist"):
        self.base_url = base_url

        load_dotenv()
        self.headers = {"User-Agent": os.getenv("MUSICBRAINZ_USERAGENT")}
        # Add "MUSICBRAINZ_USERAGENT="MusicInfluenceNetwork/1.0 (<MUSICBRAINZ_EMAIL>)" in .env
        # Not an api key, free non-commercially: https://musicbrainz.org/doc/MusicBrainz_API

    def artists_genre(self, genre: str, num_requests: int = 1000, limit: int = 100, offset: int = 0, save_file: bool = True):
        
        artists = []
        
        while len(artists) < num_requests:
            params = {
                "query": f'tag:"{genre}" AND type:"Group"',
                "fmt": "json",
                "limit": limit,
                "offset": offset
            }
            
            response = requests.get(self.base_url, headers=self.headers, params=params)
            data = response.json()
            artists_last_requests = data.get("artists", [])
            artists.extend(artists_last_requests)

            if len(artists_last_requests) < limit:
                break
            
            offset += limit
            time.sleep(1) # Ideal sleep time to keep requests passing in musicbrainz

        if (save_file):
            save_artists_to_json(artists, genre, num_requests)

        print("Musicbrainz artists loaded")
        return artists

# Optional save json file
def save_artists_to_json(artists, genre: str, num_artists: int):
    for index, artist in enumerate(artists):
        artist["index"] = index

    file_name = f"{num_artists}_{genre}.json"
    with open(file_name, "w") as file:
        json.dump(artists, file, indent=2)
    
    print(f"Data saved to {file_name}")