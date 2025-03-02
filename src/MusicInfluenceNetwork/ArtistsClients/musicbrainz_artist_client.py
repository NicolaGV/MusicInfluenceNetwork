from .artist_client import ArtistClient

import requests
import time
import json

from pathlib import Path


class MusicbrainzArtistClient(ArtistClient):

    def __init__(self, user_agent: str, base_url: str = "https://musicbrainz.org/ws/2/artist"):
        super().__init__(user_agent)
        self.base_url = base_url
        
        self.headers = {"User-Agent": self.user_agent}
        # Add "MUSICBRAINZ_USERAGENT="MusicInfluenceNetwork/1.0 (<MUSICBRAINZ_EMAIL>)" in .env
        # Not an api key, free non-commercially: https://musicbrainz.org/doc/MusicBrainz_API

    def get_single_artist_by_id(self, artist_id):

        url = f"{self.base_url}/{artist_id}"
        params = {
            "fmt": "json",
        }

        response = requests.get(url, headers=self.headers, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code}")
            return None

    def get_multiple_artist_by_id(self, artist_id_list):

        artists = []
        for artist_id in artist_id_list:
            response = self.get_single_artist_by_id(artist_id)
            if response:
                artists.append(response)
            time.sleep(1)
        return artists

    def artists_genre(self, genre: str, num_requests: int = 1000, limit: int = 100, offset: int = 0, save_file: bool = True):
        
        artists = []
        
        while len(artists) < num_requests:
            params = {
                "query": f'tag:"{genre}" AND type:"Group" OR type:"Person"',
                "fmt": "json",
                "limit": limit,
                "offset": offset
            }
            # Artist properties type: https://musicbrainz.org/doc/Artist
            
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