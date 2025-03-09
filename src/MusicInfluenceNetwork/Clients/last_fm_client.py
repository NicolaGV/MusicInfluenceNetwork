
import requests

from . import Client

from MusicInfluenceNetwork.Models import SimilarArtist


class LastFMClient(Client):
    def __init__(self, api_key: str, base_url: str = "http://ws.audioscrobbler.com/2.0/"):
        super().__init__(api_key)
        self.base_url = base_url
        
        
    def get_similar_artists(self, artist: str) -> list[SimilarArtist]:
        url = self.base_url + "?method=artist.getsimilar&artist=" + artist + "&api_key=" + self.api_key + "&format=json"
        response = requests.get(url)
        try:
            data = response.json()
        except requests.exceptions.JSONDecodeError as e:
            print(f"Invalid response for artist {artist}")
            return []
        
        similar_artists = []
        try:
            for i in range(0, len(data['similarartists']['artist'])):
                artist_data: dict = data['similarartists']['artist'][i]
                similar_artist = SimilarArtist(
                    name=artist_data.get('name'),
                    mbid=artist_data.get('mbid'), # Many artists don't have a MusicBrainz ID
                    last_fm_match=float(artist_data.get('match'))
                )
                similar_artists.append(similar_artist)
        except KeyError as e:
            print(f"Artist {artist} isn't on Last FM")
            return []
        return similar_artists



    
    
    